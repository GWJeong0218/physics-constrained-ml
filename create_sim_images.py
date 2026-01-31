"""
Synthetic SEM Image Generator (Reference Implementation)
--------------------------------------------------------

This script generates synthetic SEM-like images with
physics-inspired constraints:

- Elemental composition y satisfies:
  * non-negativity
  * sum(y) = 1
- Z-contrast proxy via effective atomic number Z_eff
- Class-specific morphology regimes (visually separable)
- SEM-style corruptions (noise, drift, scanlines, charging, etc.)
- Final tone matching to real SEM images (percentile-based)

This code is intended as a **conceptual reference implementation**,
not as a full, production-ready pipeline.
"""

import os
import csv
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ======================================================
# Global configuration (can be overridden by CLI)
# ======================================================

IMG_SIZE = 224

# Class = morphology regime (visually separable)
NUM_CLASSES = 4
IMAGES_PER_CLASS = 500

ELEMENTS = ["O", "Si", "Co", "Pd", "C"]
Z_MAP = {"O": 8, "Si": 14, "Co": 27, "Pd": 46, "C": 6}
Z_VEC = np.array([Z_MAP[e] for e in ELEMENTS], dtype=np.float32)


# ======================================================
# Tone statistics from real SEM images
# ======================================================
def compute_real_tone_stats(real_paths):
    """
    Compute percentile-based tone statistics from a set of
    real SEM reference images.

    Parameters
    ----------
    real_paths : list of str or Path
        Paths to real SEM images (grayscale).

    Returns
    -------
    dict
        Percentiles (1, 5, 50, 95, 99) as floats.
    """
    vals = []
    for p in real_paths:
        p = str(p)
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            print(f"[WARN] Failed to read real SEM: {p}")
            continue
        vals.append(im.reshape(-1))

    if len(vals) == 0:
        raise RuntimeError("No real SEM images could be read. "
                           "Check --real-ref paths.")

    vals = np.concatenate(vals, axis=0).astype(np.float32)
    p1, p5, p50, p95, p99 = np.percentile(vals, [1, 5, 50, 95, 99])
    return dict(p1=float(p1), p5=float(p5),
                p50=float(p50), p95=float(p95), p99=float(p99))


def match_real_tone(img01, tone):
    """
    Match the tone of a synthetic image to real SEM statistics.

    Parameters
    ----------
    img01 : np.ndarray, float32
        Image in [0, 1].
    tone : dict
        Percentile stats from compute_real_tone_stats().

    Returns
    -------
    np.ndarray, uint8
        Tone-matched 8-bit image.
    """
    x = np.clip(img01, 0, 1).astype(np.float32)
    x_u8 = x * 255.0

    # source percentiles
    s5, s50, s95 = np.percentile(x_u8, [5, 50, 95])
    # reference percentiles
    r5, r50, r95 = tone["p5"], tone["p50"], tone["p95"]

    # basic contrast stretching
    scale = (r95 - r5) / (s95 - s5 + 1e-6)
    y = (x_u8 - s5) * scale + r5

    # median alignment (soft)
    cur50 = np.percentile(y, 50)
    y = y + (r50 - cur50) * 0.6

    # clamp to reference extreme percentiles
    y = np.clip(y, tone["p1"], tone["p99"])
    return y.astype(np.uint8)


# ======================================================
# Composition sampler (sum = 1) with weak class bias
# ======================================================
def sample_composition(cls, base_conc=18.0, class_strength=0.10):
    base = np.ones(len(ELEMENTS), dtype=np.float32)
    weak_bias = {
        0: np.array([1.05, 1.05, 0.95, 0.95, 1.00], np.float32),
        1: np.array([0.95, 1.00, 1.05, 1.00, 1.00], np.float32),
        2: np.array([1.00, 0.95, 1.00, 1.05, 1.00], np.float32),
        3: np.array([1.05, 0.95, 0.95, 0.95, 1.10], np.float32),
    }[cls]
    bias = (1.0 - class_strength) * base + class_strength * weak_bias
    alpha = bias / bias.sum() * base_conc
    return np.random.dirichlet(alpha).astype(np.float32)


# ======================================================
# Z-contrast proxy: composition -> Z_eff -> intensity
# ======================================================
def zeff_from_comp(y, p_range=(0.6, 2.2)):
    p = np.random.uniform(*p_range)
    return float((np.sum(y * (Z_VEC ** p))) ** (1.0 / p))


def intensity_from_zeff(zeff):
    zmin, zmax = 6.0, 46.0
    x = (zeff - zmin) / (zmax - zmin + 1e-6)
    x = np.clip(x, 0, 1)

    gamma = np.random.uniform(0.7, 2.2)
    x = x ** gamma

    a = np.random.uniform(0.6, 1.5)
    b = np.random.uniform(-0.12, 0.12)
    x = np.clip(a * x + b, 0, 1)

    k = np.random.uniform(2.0, 6.0)
    x = 1.0 / (1.0 + np.exp(-k * (x - 0.5)))
    return float(np.clip(x, 0, 1))


# ======================================================
# Utility: irregular particles + rough edges
# ======================================================
def irregular_blob(canvas, cx, cy, r, n_verts=16):
    angles = np.linspace(0, 2 * np.pi, n_verts, endpoint=False)
    angles += np.random.uniform(-0.18, 0.18, size=n_verts)

    rad = r * (1.0 + np.random.normal(0, 0.20, size=n_verts))
    rad = np.clip(rad, r * 0.55, r * 1.7)

    pts = []
    for a, rr in zip(angles, rad):
        x = int(cx + rr * np.cos(a))
        y = int(cy + rr * np.sin(a))
        pts.append([np.clip(x, 0, IMG_SIZE - 1),
                    np.clip(y, 0, IMG_SIZE - 1)])
    pts = np.array([pts], dtype=np.int32)
    cv2.fillPoly(canvas, pts, 1.0)
    return canvas


def roughen_mask(mask, roughness=0.7):
    k1 = np.random.choice([3, 5, 7])
    ker1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))
    if np.random.rand() < 0.5:
        mask = cv2.erode(mask, ker1, iterations=np.random.randint(1, 3))
    else:
        mask = cv2.dilate(mask, ker1, iterations=np.random.randint(1, 3))

    g = np.random.choice([6, 8, 10])
    field = cv2.resize(np.random.randn(g, g).astype(np.float32),
                       (IMG_SIZE, IMG_SIZE),
                       interpolation=cv2.INTER_CUBIC)
    field = (field - field.mean()) / (field.std() + 1e-6)
    field = cv2.GaussianBlur(
        field, (0, 0),
        sigmaX=np.random.uniform(1.0, 2.5)
    )
    thresh = 0.5 + roughness * 0.10 * field

    out = (mask > thresh).astype(np.float32)
    if np.random.rand() < 0.8:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return np.clip(out, 0, 1)


# ======================================================
# Micro fix #1: intra-grain texture (SEM is never perfectly clean)
# ======================================================
def add_intragranular_texture(img, strength_range=(0.03, 0.06)):
    """
    Add intra-grain texture: mid/high-frequency noise + weak speckle.

    Parameters
    ----------
    img : np.ndarray, float32
        Image in [0, 1].

    Returns
    -------
    np.ndarray, float32
        Image in [0, 1] with added texture.
    """
    s = np.random.uniform(*strength_range)

    # mid-frequency noise
    n1 = np.random.randn(IMG_SIZE, IMG_SIZE).astype(np.float32)
    n1 = cv2.GaussianBlur(
        n1, (0, 0),
        sigmaX=np.random.uniform(0.6, 1.2)
    )

    # low-frequency field (weak)
    g = np.random.choice([10, 14, 18])
    n2 = cv2.resize(np.random.randn(g, g).astype(np.float32),
                    (IMG_SIZE, IMG_SIZE),
                    interpolation=cv2.INTER_CUBIC)
    n2 = cv2.GaussianBlur(
        n2, (0, 0),
        sigmaX=np.random.uniform(1.5, 3.0)
    )

    field = 0.7 * n1 + 0.3 * n2
    field = field / (field.std() + 1e-6)

    out = img * (1.0 + s * field)
    out = out + np.random.normal(
        0, s * 0.15, size=img.shape
    ).astype(np.float32)
    return np.clip(out, 0, 1)


# ======================================================
# Micro fix #2: class_3 anisotropy + streak/gradient bias
# ======================================================
def anisotropic_stretch(img, sx, sy):
    h, w = img.shape
    M = np.array([[sx, 0, (1 - sx) * w / 2],
                  [0, sy, (1 - sy) * h / 2]], dtype=np.float32)
    out = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    return np.clip(out, 0, 1)


def add_class3_streak_and_bias(
    img,
    streak_strength_range=(0.015, 0.045),
    bias_strength_range=(0.03, 0.10),
):
    """
    Simple charging-like effect: directional gradient + scanline banding.
    """
    h, w = img.shape

    # gradient bias
    bx = np.random.uniform(-1, 1)
    by = np.random.uniform(-1, 1)
    xs = np.linspace(-1, 1, w, dtype=np.float32)
    ys = np.linspace(-1, 1, h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    grad = bx * X + by * Y
    grad = grad / (np.max(np.abs(grad)) + 1e-6)
    b = np.random.uniform(*bias_strength_range)
    out = img * (1.0 + b * grad)

    # streak / banding along scan direction
    s = np.random.uniform(*streak_strength_range)
    period = np.random.randint(24, 80)
    y = np.arange(h, dtype=np.float32)
    band = np.sin(2 * np.pi * y / period) + \
        0.25 * np.random.randn(h).astype(np.float32)
    band = (band - band.mean()) / (band.std() + 1e-6)
    out = out * (1.0 + s * band[:, None])

    return np.clip(out, 0, 1)


# ======================================================
# Micro fix #3: class_2 core shear (avoid overly circular agglomerates)
# ======================================================
def shear_warp(img, shx=0.12, shy=0.0):
    h, w = img.shape
    M = np.array([[1, shx, -shx * w / 2],
                  [shy, 1, -shy * h / 2]], dtype=np.float32)
    out = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    return np.clip(out, 0, 1)


# ======================================================
# Morphology generators (visually separable regimes)
# ======================================================
def gen_fine_dense():
    img = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    n = np.random.randint(320, 520)
    for _ in range(n):
        r = np.random.randint(2, 5)
        x, y = np.random.randint(0, IMG_SIZE, 2)
        if np.random.rand() < 0.85:
            img = irregular_blob(
                img, x, y, r,
                n_verts=np.random.randint(10, 18)
            )
        else:
            cv2.circle(img, (x, y), r, 1, -1)
    img = roughen_mask(
        img,
        roughness=np.random.uniform(0.35, 0.75)
    )
    return np.clip(img, 0, 1)


def gen_coarse_growth():
    img = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    n = np.random.randint(25, 70)
    for _ in range(n):
        r = np.random.randint(10, 22)
        x, y = np.random.randint(0, IMG_SIZE, 2)
        img = irregular_blob(
            img, x, y, r,
            n_verts=np.random.randint(12, 22)
        )
    img = roughen_mask(
        img,
        roughness=np.random.uniform(0.45, 0.90)
    )
    return np.clip(img, 0, 1)


def gen_agglomerated():
    img = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    cx = IMG_SIZE // 2 + np.random.randint(-25, 25)
    cy = IMG_SIZE // 2 + np.random.randint(-25, 25)

    core_n = np.random.randint(120, 260)
    core = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)

    for _ in range(core_n):
        r = np.random.randint(3, 9)
        x = int(np.random.normal(cx, 18))
        y = int(np.random.normal(cy, 18))
        if 0 <= x < IMG_SIZE and 0 <= y < IMG_SIZE:
            core = irregular_blob(
                core, x, y, r,
                n_verts=np.random.randint(10, 18)
            )

    # core shear (micro fix #3)
    if np.random.rand() < 0.85:
        core = shear_warp(
            core,
            shx=np.random.uniform(-0.18, 0.18),
            shy=np.random.uniform(-0.05, 0.05),
        )

    # satellites
    sat_n = np.random.randint(60, 140)
    sat = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    for _ in range(sat_n):
        r = np.random.randint(2, 6)
        x, y = np.random.randint(0, IMG_SIZE, 2)
        if np.random.rand() < 0.7:
            sat = irregular_blob(
                sat, x, y, r,
                n_verts=np.random.randint(10, 16)
            )
        else:
            cv2.circle(sat, (x, y), r, 1, -1)

    img = np.maximum(core, sat)
    img = roughen_mask(
        img,
        roughness=np.random.uniform(0.55, 1.00)
    )
    return np.clip(img, 0, 1)


def gen_sparse_charged():
    img = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    n = np.random.randint(15, 50)
    for _ in range(n):
        r = np.random.randint(6, 14)
        x, y = np.random.randint(0, IMG_SIZE, 2)
        img = irregular_blob(
            img, x, y, r,
            n_verts=np.random.randint(10, 18)
        )
    img = roughen_mask(
        img,
        roughness=np.random.uniform(0.30, 0.80)
    )
    return np.clip(img, 0, 1)


GEN_BY_CLASS = {
    0: gen_fine_dense,
    1: gen_coarse_growth,
    2: gen_agglomerated,
    3: gen_sparse_charged,
}


# ======================================================
# Phase map / composition perturbation
# ======================================================
def make_precip_mask(num, r_range):
    m = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    for _ in range(num):
        r = np.random.randint(r_range[0], r_range[1])
        x, y = np.random.randint(0, IMG_SIZE, 2)
        cv2.circle(m, (x, y), r, 1, -1)
    m = cv2.GaussianBlur(m, (5, 5), 0)
    return np.clip(m, 0, 1)


def perturb_composition(y, sigma):
    y2 = y + np.random.normal(0, sigma, size=y.shape).astype(np.float32)
    y2 = np.clip(y2, 1e-4, None)
    y2 = y2 / y2.sum()
    return y2


# ======================================================
# SEM imaging pipeline
# ======================================================
def add_sem_background(
    img,
    base_range=(0.06, 0.18),
    lf_strength_range=(0.03, 0.12),
):
    h, w = img.shape
    base = np.random.uniform(*base_range)
    out = img + base

    g = int(np.random.choice([6, 8, 10]))
    small = np.random.randn(g, g).astype(np.float32)
    field = cv2.resize(
        small, (w, h),
        interpolation=cv2.INTER_CUBIC
    )
    field = (field - field.mean()) / (field.std() + 1e-6)
    out = out + np.random.uniform(*lf_strength_range) * field

    out = out + np.random.normal(
        0, 0.01, size=(h, w)
    ).astype(np.float32)
    return np.clip(out, 0, 1)


def apply_blur(img, prob):
    if np.random.rand() < prob:
        k = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img


def apply_intensity_variation(img, strength):
    alpha = np.random.uniform(1.0 - strength, 1.0 + strength)
    beta = np.random.uniform(-0.2 * strength, 0.2 * strength)
    return np.clip(alpha * img + beta, 0, 1)


def apply_shot_noise(img, exposure_range):
    exposure = np.random.uniform(*exposure_range)
    lam = np.clip(img, 0, 1) * exposure
    noisy = np.random.poisson(lam).astype(np.float32) / exposure
    return np.clip(noisy, 0, 1)


def apply_lowfreq_gradient(img, strength, grid):
    small = np.random.randn(grid, grid).astype(np.float32)
    field = cv2.resize(
        small, (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_CUBIC
    )
    field = (field - field.min()) / (field.max() - field.min() + 1e-6)
    field = (field - 0.5) * 2.0
    scale = 1.0 + strength * field
    out = img * scale
    out += np.random.uniform(-0.05, 0.05) * strength
    return np.clip(out, 0, 1)


def apply_drift(img, max_shift):
    dx = np.random.uniform(-max_shift, max_shift)
    dy = np.random.uniform(-max_shift, max_shift)
    M = np.array([[1, 0, dx],
                  [0, 1, dy]], dtype=np.float32)
    out = cv2.warpAffine(
        img, M, (IMG_SIZE, IMG_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    return np.clip(out, 0, 1)


def apply_scanline(img, strength, period_range=(20, 70)):
    period = np.random.randint(period_range[0], period_range[1] + 1)
    y = np.arange(IMG_SIZE, dtype=np.float32)
    band = np.sin(2.0 * np.pi * y / period)
    band = band + 0.3 * np.random.randn(IMG_SIZE).astype(np.float32)
    band = (band - band.min()) / (band.max() - band.min() + 1e-6)
    band = (band - 0.5) * 2.0
    band = 1.0 + strength * band
    out = img * band[:, None]
    return np.clip(out, 0, 1)


def apply_partial_occlusion(img, max_occ_ratio=0.18):
    h, w = img.shape
    occ_h = int(np.random.uniform(0.06, max_occ_ratio) * h)
    occ_w = int(np.random.uniform(0.06, max_occ_ratio) * w)
    x0 = np.random.randint(0, w - occ_w)
    y0 = np.random.randint(0, h - occ_h)

    mask = np.zeros((h, w), np.float32)
    mask[y0:y0 + occ_h, x0:x0 + occ_w] = 1.0
    k = np.random.choice([9, 13, 17])
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = np.clip(mask, 0, 1)

    bg = float(np.percentile(img, 50))
    bg = max(bg, 0.05)
    dark = np.random.uniform(0.25, 0.55)
    stain = img * dark + bg * np.random.uniform(0.8, 1.2)
    stain += np.random.normal(
        0, 0.01, size=img.shape
    ).astype(np.float32)
    stain = np.clip(stain, 0, 1)

    out = img * (1.0 - mask) + stain * mask
    return np.clip(out, 0, 1)


def apply_scale_variation(img, min_scale=0.85, max_scale=1.15):
    scale = np.random.uniform(min_scale, max_scale)
    h, w = img.shape
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    new_h = max(2, new_h)
    new_w = max(2, new_w)
    img_scaled = cv2.resize(
        img, (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    if scale < 1.0:
        pad_top = (h - new_h) // 2
        pad_bottom = h - new_h - pad_top
        pad_left = (w - new_w) // 2
        pad_right = w - new_w - pad_left
        out = cv2.copyMakeBorder(
            img_scaled, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT_101
        )
        return np.clip(out[:h, :w], 0, 1)
    else:
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        out = img_scaled[start_h:start_h + h, start_w:start_w + w]
        return np.clip(out, 0, 1)


# ======================================================
# Class-specific imaging parameters
# ======================================================
CLASS_PARAMS = {
    0: dict(blur_prob=0.25, intensity=0.10, exposure=(90, 240),
            drift_p=0.20, scan_p=0.20, scan_s=(0.01, 0.04)),
    1: dict(blur_prob=0.35, intensity=0.16, exposure=(80, 220),
            drift_p=0.25, scan_p=0.25, scan_s=(0.01, 0.05)),
    2: dict(blur_prob=0.45, intensity=0.20, exposure=(70, 200),
            drift_p=0.30, scan_p=0.30, scan_s=(0.02, 0.06)),
    3: dict(blur_prob=0.55, intensity=0.26, exposure=(55, 170),
            drift_p=0.55, scan_p=0.60, scan_s=(0.03, 0.09)),
}


# ======================================================
# Main generation routine
# ======================================================
def generate_dataset(
    out_dir: Path,
    real_refs,
    num_classes=NUM_CLASSES,
    images_per_class=IMAGES_PER_CLASS,
    seed=42,
):
    np.random.seed(seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    for c in range(num_classes):
        (out_dir / f"class_{c}").mkdir(exist_ok=True)

    labels_csv = out_dir / "labels.csv"

    # compute real SEM tone
    real_refs = [Path(p) for p in real_refs]
    real_tone = compute_real_tone_stats(real_refs)
    print("[INFO] Real SEM tone statistics:", real_tone)

    print("[INFO] Generating synthetic SEM images")

    with labels_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path"] + ELEMENTS)

        for cls in range(num_classes):
            gen = GEN_BY_CLASS[cls]
            params = CLASS_PARAMS[cls]

            for i in tqdm(range(images_per_class),
                          desc=f"class_{cls}"):
                # 0) composition
                y = sample_composition(
                    cls,
                    base_conc=np.random.uniform(14.0, 26.0),
                    class_strength=0.10,
                )

                # 1) morphology
                base = gen()

                # 2) phase map (class-dependent)
                if cls == 0:
                    mask = make_precip_mask(
                        num=np.random.randint(30, 120),
                        r_range=(2, 7),
                    )
                    sigma = np.random.uniform(0.015, 0.04)
                elif cls == 1:
                    mask = make_precip_mask(
                        num=np.random.randint(10, 60),
                        r_range=(4, 12),
                    )
                    sigma = np.random.uniform(0.015, 0.05)
                elif cls == 2:
                    mask = make_precip_mask(
                        num=np.random.randint(20, 100),
                        r_range=(3, 10),
                    )
                    sigma = np.random.uniform(0.020, 0.060)
                else:
                    mask = make_precip_mask(
                        num=np.random.randint(8, 40),
                        r_range=(5, 14),
                    )
                    sigma = np.random.uniform(0.015, 0.045)

                y2 = perturb_composition(y, sigma=sigma)

                # 3) Z-contrast
                i1 = intensity_from_zeff(zeff_from_comp(y))
                i2 = intensity_from_zeff(zeff_from_comp(y2))
                img = base * ((1.0 - mask) * i1 + mask * i2)

                # 4) background
                img = add_sem_background(img)

                # 5) corruptions
                img = apply_blur(img, prob=params["blur_prob"])
                img = apply_intensity_variation(
                    img, strength=params["intensity"]
                )
                img = apply_shot_noise(
                    img, exposure_range=params["exposure"]
                )

                # 6) low-frequency multiplicative field
                img = apply_lowfreq_gradient(
                    img,
                    strength=np.random.uniform(0.04, 0.14),
                    grid=int(np.random.choice([6, 8, 10])),
                )

                # 7) drift / scanline
                if np.random.rand() < params["drift_p"]:
                    img = apply_drift(
                        img,
                        max_shift=np.random.uniform(0.5, 2.5),
                    )
                if np.random.rand() < params["scan_p"]:
                    img = apply_scanline(
                        img,
                        strength=np.random.uniform(*params["scan_s"]),
                        period_range=(20, 70),
                    )

                # 8) occlusion / scale variation
                if np.random.rand() < 0.35:
                    img = apply_partial_occlusion(
                        img, max_occ_ratio=0.18
                    )
                if np.random.rand() < 0.60:
                    img = apply_scale_variation(
                        img,
                        min_scale=0.85,
                        max_scale=1.15,
                    )

                # 9) intra-grain texture (micro fix #1)
                img = add_intragranular_texture(
                    img, strength_range=(0.03, 0.06)
                )

                # 10) class_3 charging-like artifacts (micro fix #2)
                if cls == 3:
                    sx = np.random.uniform(1.05, 1.15)
                    sy = np.random.uniform(0.92, 0.98)
                    if np.random.rand() < 0.5:
                        sx, sy = sy, sx
                    img = anisotropic_stretch(img, sx=sx, sy=sy)
                    img = add_class3_streak_and_bias(img)

                # 11) final tone match to real SEM
                img_u8 = match_real_tone(img, real_tone)
                img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

                out_path = out_dir / f"class_{cls}" / f"sim_{cls}_{i:04d}.png"
                cv2.imwrite(str(out_path), img_bgr)

                writer.writerow([str(out_path)] + [float(v) for v in y])

    print("[INFO] Generation finished")
    print(f"[INFO] Images saved to: {out_dir}")
    print(f"[INFO] Labels saved to: {labels_csv}")


# ======================================================
# CLI entry point
# ======================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthetic SEM image generator "
                    "(physics-constrained reference implementation)."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="sim_images",
        help="Output directory for images and labels.csv "
             "(default: sim_images)",
    )
    parser.add_argument(
        "--real-ref",
        type=str,
        nargs="+",
        required=True,
        help="Paths to real SEM reference images "
             "(at least one, grayscale or convertible).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        help=f"Number of morphology classes (default: {NUM_CLASSES})",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=IMAGES_PER_CLASS,
        help=f"Images per class (default: {IMAGES_PER_CLASS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(
        out_dir=Path(args.out_dir),
        real_refs=args.real_ref,
        num_classes=args.num_classes,
        images_per_class=args.images_per_class,
        seed=args.seed,
    )
