"""
Leave-One-Image-Out (LOIO) Runner
---------------------------------

Reference implementation for the LOIO experiments in:

    "Physics-Constrained Deep Learning for SEM Images
     under Extreme Few-Shot Conditions"

Key ideas reflected in this script:

- Extreme few-shot regime (N images, LOIO)
- Physics-constrained output space:
    * non-negative composition
    * sum-to-1 (simplex renormalization)
    * optional hard masking for absent elements
    * optional leak penalty on absent-element mass
- Sim-to-real backbone:
    * EfficientNet-B0 pretrained on physics-based synthetic SEM (optional)
- Epoch-aware random train patches, deterministic val/test patches

This script is intended as a *conceptual reference implementation*.
It is not a full, production-grade training pipeline, and some
hyperparameters or engineering details may differ from the paper's
final configuration.
"""

import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import argparse

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ============================================================
# Configuration
# ============================================================

@dataclass
class CFG:
    # device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # data paths
    img_dir: str = "./images"
    labels_xlsx: str = "./labels.xlsx"

    # labels.xlsx columns
    img_col: str = "filename"
    elements: List[str] = None          # will be set after init
    has_prefix: str = "has_"

    # Stage2 backbone (SIM) - backbone state_dict only
    stage2_backbone_path: str = "./stage2_backbone.pth"

    # patching
    patch_size: int = 224
    patches_per_image_train: int = 64
    patches_per_image_val: int = 24
    patches_per_image_test: int = 96

    # optimization
    batch_size: int = 64
    epochs: int = 100
    lr_head: float = 2e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-2
    early_patience: int = 5

    # real fine-tune unfreeze depth (0 = backbone fully frozen)
    real_unfreeze_depth: int = 0

    # HardMask applied to composition
    # This variant keeps training unconstrained, and applies
    # GT-based hard masking only at evaluation time by default.
    apply_gt_hardmask_train: bool = False
    apply_gt_hardmask_eval: bool = True

    # leak penalty (suppresses probability mass on absent elements)
    lambda_leak: float = 0.0

    # misc
    seed: int = 42
    num_workers: int = 0

    # test forward chunk size (speed)
    test_forward_bs: int = 64

    # where to save LOIO summaries and fold-level CSVs
    out_dir: str = "./loio_results"


# default element order (matches the paper)
DEFAULT_ELEMENTS = ["O", "Si", "Co", "Pd", "C"]


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def normalize_0_1(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)


def deterministic_crop_coords(h: int, w: int, crop: int, key: int) -> Tuple[int, int]:
    rng = np.random.RandomState(key)
    y = 0 if h <= crop else int(rng.randint(0, h - crop + 1))
    x = 0 if w <= crop else int(rng.randint(0, w - crop + 1))
    return y, x


def random_crop_coords(h: int, w: int, crop: int, rng: np.random.RandomState) -> Tuple[int, int]:
    y = 0 if h <= crop else int(rng.randint(0, h - crop + 1))
    x = 0 if w <= crop else int(rng.randint(0, w - crop + 1))
    return y, x


def make_patch(img: np.ndarray, y: int, x: int, crop: int) -> np.ndarray:
    h, w = img.shape
    if h < crop or w < crop:
        pad_h = max(0, crop - h)
        pad_w = max(0, crop - w)
        img = np.pad(
            img,
            ((pad_h // 2, pad_h - pad_h // 2),
             (pad_w // 2, pad_w - pad_w // 2)),
            mode="reflect",
        )
        h, w = img.shape

    patch = img[y:y + crop, x:x + crop]
    if patch.shape[0] != crop or patch.shape[1] != crop:
        patch = cv2.resize(patch, (crop, crop), interpolation=cv2.INTER_LINEAR)
    return patch


def renorm_simplex(x: torch.Tensor) -> torch.Tensor:
    return x / (x.sum(dim=1, keepdim=True) + 1e-8)


def hardmask_with_mask(pred: torch.Tensor, mask01: torch.Tensor) -> torch.Tensor:
    """
    v4 hardmask logic:
    - multiply by mask
    - renormalize to simplex
    - multiply again (suppress leakage)
    - renormalize again
    """
    out = pred * mask01
    out = renorm_simplex(out)
    out = out * mask01
    out = renorm_simplex(out)
    return out


def hardmask_numpy(pred: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    out = pred * mask01
    s = out.sum(axis=-1, keepdims=True) + 1e-8
    out = out / s
    out = out * mask01
    s = out.sum(axis=-1, keepdims=True) + 1e-8
    out = out / s
    return out.astype(np.float32)


# ============================================================
# Load labels (composition + has_*)
# ============================================================

def load_items(cfg: CFG) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Load image paths, compositions, and presence masks from labels.xlsx.

    Expected columns:
        - cfg.img_col: relative or absolute path to the image file
        - one column per element in cfg.elements (compositions)
        - one column per element with prefix cfg.has_prefix (presence mask)
    """
    df = pd.read_excel(cfg.labels_xlsx)

    if cfg.img_col not in df.columns:
        raise ValueError(
            f"Column '{cfg.img_col}' not found in labels file. "
            f"Available columns: {list(df.columns)}"
        )

    paths = df[cfg.img_col].astype(str).tolist()
    fixed = [p if os.path.isabs(p) else os.path.join(cfg.img_dir, p) for p in paths]
    df["__imgpath__"] = fixed

    # compositions
    for e in cfg.elements:
        if e not in df.columns:
            raise ValueError(
                f"Element column '{e}' not found. "
                f"Available columns: {list(df.columns)}"
            )

    # presence indicators
    has_cols = [cfg.has_prefix + e for e in cfg.elements]
    for c in has_cols:
        if c not in df.columns:
            raise ValueError(
                f"Presence column '{c}' not found. "
                f"Available columns: {list(df.columns)}"
            )

    items = []
    for _, r in df.iterrows():
        p = str(r["__imgpath__"])

        y = r[cfg.elements].values.astype(np.float32)
        y = y / (float(y.sum()) + 1e-8)

        has = r[has_cols].values.astype(np.float32)
        has = (has > 0.5).astype(np.float32)

        # safety: if has=0, force y=0 for that element
        y = y * has
        y = y / (float(y.sum()) + 1e-8)

        items.append((p, y, has))

    if len(items) < 3:
        raise ValueError("LOIO requires at least 3 images.")
    return items


# ============================================================
# Dataset (epoch-aware train random crop)
# ============================================================

class PatchDataset(Dataset):
    """
    Patch-level dataset.

    - Train split: epoch-aware random patching (reproducible)
    - Val/Test splits: deterministic patching
    """

    def __init__(self, items: List[Tuple[str, np.ndarray, np.ndarray]], split: str, cfg: CFG):
        assert split in {"train", "val", "test"}
        self.items = items
        self.split = split
        self.cfg = cfg
        self.epoch = 0

        if split == "train":
            self.ppi = cfg.patches_per_image_train
        elif split == "val":
            self.ppi = cfg.patches_per_image_val
        else:
            self.ppi = cfg.patches_per_image_test

        self.index = []
        for i in range(len(items)):
            for p in range(self.ppi):
                self.index.append((i, p))

        w = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.mean = torch.tensor(w.transforms().mean).view(3, 1, 1)
        self.std = torch.tensor(w.transforms().std).view(3, 1, 1)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        img_i, p_i = self.index[idx]
        path, y, has = self.items[img_i]

        img = normalize_0_1(read_gray(path))

        if self.split == "train":
            # epoch-aware random patches (reproducible)
            key = (
                self.cfg.seed
                + 1_000_000
                + self.epoch * 100_000
                + img_i * 1000
                + p_i
            )
            rng = np.random.RandomState(key)
            y0, x0 = random_crop_coords(
                img.shape[0], img.shape[1], self.cfg.patch_size, rng
            )
        else:
            # val/test: deterministic cropping
            split_bias = {"val": 200_000, "test": 300_000}[self.split]
            key = self.cfg.seed + split_bias + img_i * 1000 + p_i
            y0, x0 = deterministic_crop_coords(
                img.shape[0], img.shape[1], self.cfg.patch_size, key
            )

        patch = make_patch(img, y0, x0, self.cfg.patch_size)

        patch = torch.from_numpy(patch).unsqueeze(0).float()  # (1,H,W)
        patch = patch.repeat(3, 1, 1)                         # (3,H,W)

        # ImageNet normalization
        patch = (patch - self.mean) / (self.std + 1e-12)

        y_t = torch.from_numpy(y).float()
        has_t = torch.from_numpy(has).float()
        return patch, y_t, has_t


# ============================================================
# Model: composition-only head on EfficientNet-B0
# ============================================================

class CompositionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim),
        )
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, feat):
        x = self.softplus(self.mlp(feat))
        x = renorm_simplex(x)
        return x


class Model(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.comp_head = CompositionHead(in_dim, out_dim)

    def forward(self, x):
        feat = self.backbone(x)
        comp = self.comp_head(feat)
        return comp


def freeze_all(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False


def unfreeze_last_stages_efficientnet(model: Model, depth: int):
    """
    Freeze all backbone layers, then optionally unfreeze the last `depth` feature blocks.
    """
    freeze_all(model.backbone)
    if depth <= 0:
        return
    feats = model.backbone.features
    n = len(feats)
    for i in range(max(0, n - depth), n):
        for p in feats[i].parameters():
            p.requires_grad = True


def load_stage2_backbone(model: Model, backbone_path: str):
    """
    Load a SIM-pretrained EfficientNet-B0 backbone state dict.

    Only the backbone is overwritten; the composition head remains fresh.
    """
    sd = torch.load(backbone_path, map_location="cpu")
    missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
    return missing, unexpected


def get_trainable_params(model: Model, cfg: CFG):
    head_params = [p for p in model.comp_head.parameters() if p.requires_grad]
    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = []
    if bb_params:
        groups.append({"params": bb_params, "lr": cfg.lr_backbone})
    groups.append({"params": head_params, "lr": cfg.lr_head})
    return groups


# ============================================================
# Train / Eval (MAE v4)
# ============================================================

@torch.no_grad()
def eval_mae_patch_level(model: Model, loader: DataLoader, cfg: CFG) -> float:
    """
    Patch-level MAE (v4):
        mean over elements of |pred - y|
    """
    model.eval()
    maes = []
    for x, y, has in loader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        has = has.to(cfg.device)

        comp = model(x)
        if cfg.apply_gt_hardmask_eval:
            comp = hardmask_with_mask(comp, has)

        err = (comp - y).abs().mean(dim=1)
        maes.append(err.detach().cpu().numpy())

    maes = np.concatenate(maes, axis=0)
    return float(maes.mean())


def train_one(
    model: Model,
    train_ds: PatchDataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: CFG,
) -> Dict:
    """
    Train on (N-1) images for one LOIO split.
    """
    model.to(cfg.device)

    l1 = nn.L1Loss(reduction="mean")
    opt = optim.AdamW(
        get_trainable_params(model, cfg),
        weight_decay=cfg.weight_decay,
    )

    best_val = 1e9
    best_sd = None
    patience = 0
    epochs_ran = 0

    for epoch in range(cfg.epochs):
        # epoch-aware random patching
        train_ds.set_epoch(epoch)

        model.train()

        # FIX: if backbone is fully frozen, also keep BN stats frozen
        if cfg.real_unfreeze_depth == 0:
            model.backbone.eval()

        for x, y, has in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            has = has.to(cfg.device)

            opt.zero_grad(set_to_none=True)
            pred = model(x)  # (B,E) simplex

            # train-time hardmask (usually disabled in this variant)
            if cfg.apply_gt_hardmask_train:
                pred_for_loss = hardmask_with_mask(pred, has)
            else:
                pred_for_loss = pred

            loss_comp = l1(pred_for_loss, y)

            # leak penalty on absent elements (can be disabled via lambda_leak=0)
            leak = (pred * (1.0 - has)).sum(dim=1).mean()
            loss = loss_comp + cfg.lambda_leak * leak

            loss.backward()
            opt.step()

        val_mae = eval_mae_patch_level(model, val_loader, cfg)
        epochs_ran = epoch + 1

        if val_mae < best_val - 1e-6:
            best_val = val_mae
            best_sd = {
                "backbone": {k: v.cpu() for k, v in model.backbone.state_dict().items()},
                "comp_head": {k: v.cpu() for k, v in model.comp_head.state_dict().items()},
            }
            patience = 0
        else:
            patience += 1

        if patience >= cfg.early_patience:
            break

    # restore best validation snapshot
    if best_sd is not None:
        model.backbone.load_state_dict(best_sd["backbone"], strict=False)
        model.comp_head.load_state_dict(best_sd["comp_head"], strict=True)

    return {"best_val_mae": float(best_val), "epochs_ran": int(epochs_ran)}


@torch.no_grad()
def predict_image_level(
    model: Model,
    img_path: str,
    y_true: np.ndarray,
    has_true: np.ndarray,
    cfg: CFG,
    fold_idx: int,
) -> Dict:
    """
    LOIO evaluation at image-level:
        - multiple deterministic patches
        - average predictions
        - MAE v4 on averaged composition
    """
    device = cfg.device
    model.eval()

    img = normalize_0_1(read_gray(img_path))

    patches = []
    for p_i in range(cfg.patches_per_image_test):
        key = cfg.seed + 900_000 + fold_idx * 10_000 + p_i
        y0, x0 = deterministic_crop_coords(
            img.shape[0], img.shape[1], cfg.patch_size, key
        )
        patch = make_patch(img, y0, x0, cfg.patch_size)
        patches.append(patch)

    patches = np.stack(patches, axis=0).astype(np.float32)  # (N,H,W)
    x = torch.from_numpy(patches).unsqueeze(1)              # (N,1,H,W)
    x = x.repeat(1, 3, 1, 1)                                # (N,3,H,W)

    w = EfficientNet_B0_Weights.IMAGENET1K_V1
    mean = torch.tensor(w.transforms().mean).view(1, 3, 1, 1)
    std = torch.tensor(w.transforms().std).view(1, 3, 1, 1)
    x = (x - mean) / (std + 1e-12)

    comp_list = []
    N = x.shape[0]
    for s in range(0, N, cfg.test_forward_bs):
        xb = x[s:s + cfg.test_forward_bs].to(device)
        comp = model(xb)
        comp_list.append(comp.detach().cpu().numpy())

    comp_all = np.concatenate(comp_list, axis=0)  # (N,E)
    pred_mean = comp_all.mean(axis=0).astype(np.float32)

    if cfg.apply_gt_hardmask_eval:
        pred_mean = hardmask_numpy(
            pred_mean[None, :],
            has_true[None, :],
        )[0]

    err = np.abs(pred_mean - y_true).astype(np.float32)
    img_mae = float(err.mean())  # MAE v4

    return {"img_mae": img_mae, "err": err, "pred_mean": pred_mean}


# ============================================================
# LOIO runner
# ============================================================

def run_loio(cfg: CFG, condition: str):
    """
    condition in {"nosim", "sim"}:
        - "nosim": ImageNet-pretrained backbone only
        - "sim"  : load SIM-pretrained backbone and fine-tune on real
    """
    assert condition in ["nosim", "sim"]

    items = load_items(cfg)
    n = len(items)

    use_sim_backbone = (condition == "sim")

    fold_rows = []
    maes_all = []
    elem_errs_all = []

    for leave_idx in range(n):
        test_item = items[leave_idx]
        train_items = [items[i] for i in range(n) if i != leave_idx]

        train_ds = PatchDataset(train_items, "train", cfg)
        val_ds = PatchDataset(train_items, "val", cfg)

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        model = Model(out_dim=len(cfg.elements))

        if use_sim_backbone:
            missing, unexpected = load_stage2_backbone(model, cfg.stage2_backbone_path)
            if (len(missing) + len(unexpected)) != 0:
                print(
                    f"[WARN] Backbone load issues: "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )

        unfreeze_last_stages_efficientnet(model, cfg.real_unfreeze_depth)
        for p in model.comp_head.parameters():
            p.requires_grad = True

        info = train_one(model, train_ds, train_loader, val_loader, cfg)

        model.to(cfg.device)
        test_path, y_true, has_true = test_item
        out = predict_image_level(
            model, test_path, y_true, has_true, cfg, fold_idx=leave_idx
        )

        img_mae = out["img_mae"]
        elem_err = out["err"]
        pred_mean = out["pred_mean"]

        maes_all.append(img_mae)
        elem_errs_all.append(elem_err)

        fold_rows.append(
            {
                "condition": condition,
                "leave_idx": leave_idx,
                "test_path": test_path,
                "best_val_mae": info["best_val_mae"],
                "epochs_ran": info["epochs_ran"],
                "img_mae": img_mae,
                **{f"y_{e}": float(y_true[i]) for i, e in enumerate(cfg.elements)},
                **{
                    f"pred_{e}": float(pred_mean[i])
                    for i, e in enumerate(cfg.elements)
                },
                **{
                    f"has_{e}": float(has_true[i])
                    for i, e in enumerate(cfg.elements)
                },
                **{
                    f"err_{e}": float(elem_err[i])
                    for i, e in enumerate(cfg.elements)
                },
            }
        )

        print(
            f"{condition.upper()} LOIO {leave_idx + 1}/{n} | "
            f"img_mae={img_mae:.6f} | "
            f"val_best={info['best_val_mae']:.6f} | "
            f"epochs={info['epochs_ran']}"
        )

    maes_all = np.array(maes_all, dtype=np.float32)
    elem_errs_all = np.stack(elem_errs_all, axis=0)
    elem_mae_all = elem_errs_all.mean(axis=0)

    summary = {
        "condition": condition,
        "loio_mean_mae_all": float(maes_all.mean()),
        "loio_std_mae_all": float(maes_all.std(ddof=0)),
    }
    for i, e in enumerate(cfg.elements):
        summary[f"loio_mae_{e}"] = float(elem_mae_all[i])

    return summary, fold_rows


# ============================================================
# CLI entry point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "LOIO runner (MAE v4, epoch-random train patches, "
            "optional hard masking and leak penalty)."
        )
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default="./images",
        help="Directory containing real SEM images.",
    )
    parser.add_argument(
        "--labels-xlsx",
        type=str,
        default="./labels.xlsx",
        help="Excel file with compositions and presence indicators.",
    )
    parser.add_argument(
        "--stage2-backbone",
        type=str,
        default="./stage2_backbone.pth",
        help="Path to SIM-pretrained EfficientNet-B0 backbone state dict.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./loio_results",
        help="Output directory for LOIO CSV summaries.",
    )
    parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        default=DEFAULT_ELEMENTS,
        help="Element names (order defines output dimension).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for patch-level training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of epochs per LOIO split.",
    )
    parser.add_argument(
        "--real-unfreeze-depth",
        type=int,
        default=0,
        help="Number of EfficientNet feature blocks to unfreeze from the end (0 = fully frozen backbone).",
    )
    parser.add_argument(
        "--lambda-leak",
        type=float,
        default=0.0,
        help="Weight for leak penalty on absent elements (0 disables it).",
    )
    parser.add_argument(
        "--apply-train-hardmask",
        action="store_true",
        help="If set, apply GT hardmask also during training loss.",
    )
    parser.add_argument(
        "--no-eval-hardmask",
        action="store_true",
        help="If set, disable GT hardmask at evaluation (not recommended).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = CFG()
    cfg.img_dir = args.img_dir
    cfg.labels_xlsx = args.labels_xlsx
    cfg.stage2_backbone_path = args.stage2_backbone
    cfg.out_dir = args.out_dir
    cfg.elements = args.elements
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.real_unfreeze_depth = args.real_unfreeze_depth
    cfg.lambda_leak = args.lambda_leak
    cfg.apply_gt_hardmask_train = args.apply_train_hardmask
    cfg.apply_gt_hardmask_eval = not args.no_eval_hardmask
    cfg.seed = args.seed

    set_seed(cfg.seed)

    print("[INFO] Device:", cfg.device)
    print("[INFO] Image dir:", cfg.img_dir)
    print("[INFO] Labels file:", cfg.labels_xlsx)
    print("[INFO] Stage2 backbone (SIM):", cfg.stage2_backbone_path)
    print("[INFO] Elements:", cfg.elements)
    print("[INFO] Real unfreeze depth:", cfg.real_unfreeze_depth)
    print(
        f"[INFO] GT HardMask train={cfg.apply_gt_hardmask_train} "
        f"eval={cfg.apply_gt_hardmask_eval}"
    )
    print("[INFO] lambda_leak:", cfg.lambda_leak)
    print()

    os.makedirs(cfg.out_dir, exist_ok=True)

    # LOIO without SIM backbone
    s1, f1 = run_loio(cfg, "nosim")
    print("\n--- No-SIM Summary ---")
    print(s1)

    # LOIO with SIM backbone (if available)
    s2, f2 = run_loio(cfg, "sim")
    print("\n--- SIM(Stage2) Summary ---")
    print(s2)

    summary_csv = os.path.join(
        cfg.out_dir,
        "loio_MAEv4_randpatch_leakpen_trainNoHM_summary.csv",
    )
    folds_csv = os.path.join(
        cfg.out_dir,
        "loio_MAEv4_randpatch_leakpen_trainNoHM_folds.csv",
    )

    pd.DataFrame([s1, s2]).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(f1 + f2).to_csv(folds_csv, index=False, encoding="utf-8-sig")

    print("\n[INFO] Saved:")
    print("   ", summary_csv)
    print("   ", folds_csv)

    print("\n" + "=" * 70)
    print("FINAL LOIO MAE COMPARISON (MAE=v4, epoch-random train patches)")
    print("=" * 70)
    print(
        f"No-SIM (HardMask eval only): "
        f"{s1['loio_mean_mae_all']:.6f} (std {s1['loio_std_mae_all']:.6f})"
    )
    print(
        f"SIM(Stage2)               : "
        f"{s2['loio_mean_mae_all']:.6f} (std {s2['loio_std_mae_all']:.6f})"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
