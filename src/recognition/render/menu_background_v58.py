"""Procedural menu-like background generator and linear-sRGB core contrast auditor for V58.

Six balanced background families:
- warm_offwhite_paper (light)
- dark_chalkboard (dark)
- pale_wood (light)
- dark_wood (dark)
- soft_mottled_card (light/dark)
- photographic_gradient (light/dark)

Three texture tiers: plain (25%), light_texture (50%), moderate_texture (25%).
"""
from __future__ import annotations

import random
import numpy as np
from PIL import Image

BG_FAMILIES = (
    "warm_offwhite_paper",
    "dark_chalkboard",
    "pale_wood",
    "dark_wood",
    "soft_mottled_card",
    "photographic_gradient",
)
BG_TIERS = ("plain", "light_texture", "moderate_texture")

DARK_INKS = [
    (18, 18, 20),
    (36, 26, 22),
    (22, 30, 48),
    (82, 26, 24),
    (24, 48, 34),
]
LIGHT_INKS = [
    (244, 241, 234),
    (250, 244, 218),
    (230, 238, 246),
    (238, 222, 182),
]


def srgb_to_luminance(rgb_u8: np.ndarray) -> np.ndarray:
    """Compute WCAG linear-sRGB relative luminance for (..., 3) uint8/float array."""
    c = np.clip(rgb_u8.astype(np.float32) / 255.0, 0.0, 1.0)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def contrast_ratio_map(fg_rgb: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    l1 = srgb_to_luminance(fg_rgb)
    l2 = srgb_to_luminance(bg_rgb)
    hi = np.maximum(l1, l2)
    lo = np.minimum(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _upsample_noise(np_rng: np.random.Generator, gh: int, gw: int, h: int, w: int) -> np.ndarray:
    small = np_rng.normal(0.0, 1.0, size=(max(2, gh), max(2, gw))).astype(np.float32)
    im = Image.fromarray(small, mode="F").resize((w, h), Image.Resampling.BICUBIC)
    return np.asarray(im, dtype=np.float32)


def generate_background(
    family: str,
    tier: str,
    height: int,
    width: int,
    rng: random.Random,
    np_rng: np.random.Generator,
    amp_scale: float = 1.0,
) -> tuple[np.ndarray, str]:
    """Generate a coherent (height, width, 3) uint8 procedural menu-like background and polarity ('light'|'dark')."""
    assert family in BG_FAMILIES
    assert tier in BG_TIERS

    if family == "warm_offwhite_paper":
        polarity = "light"
        base = np.array([rng.randint(232, 248), rng.randint(224, 242), rng.randint(206, 230)], dtype=np.float32)
    elif family == "dark_chalkboard":
        polarity = "dark"
        base = np.array([rng.randint(24, 44), rng.randint(28, 48), rng.randint(28, 46)], dtype=np.float32)
    elif family == "pale_wood":
        polarity = "light"
        base = np.array([rng.randint(212, 236), rng.randint(188, 216), rng.randint(152, 186)], dtype=np.float32)
    elif family == "dark_wood":
        polarity = "dark"
        base = np.array([rng.randint(42, 72), rng.randint(28, 50), rng.randint(20, 36)], dtype=np.float32)
    elif family == "soft_mottled_card":
        if rng.random() < 0.5:
            polarity = "light"
            pal = [(232, 224, 208), (218, 228, 218), (236, 220, 218), (222, 228, 236)]
        else:
            polarity = "dark"
            pal = [(38, 46, 56), (48, 36, 40), (34, 48, 42), (50, 44, 36)]
        b0 = rng.choice(pal)
        base = np.array([b0[0] + rng.randint(-6, 6), b0[1] + rng.randint(-6, 6), b0[2] + rng.randint(-6, 6)], dtype=np.float32)
    else:  # photographic_gradient
        if rng.random() < 0.5:
            polarity = "light"
            base = np.array([rng.randint(216, 244), rng.randint(212, 240), rng.randint(202, 234)], dtype=np.float32)
        else:
            polarity = "dark"
            base = np.array([rng.randint(28, 56), rng.randint(28, 56), rng.randint(32, 60)], dtype=np.float32)

    canvas = np.broadcast_to(base.reshape(1, 1, 3), (height, width, 3)).copy()
    if tier == "plain":
        grain = np_rng.normal(0.0, 1.2, size=(height, width, 1)).astype(np.float32)
        return np.clip(canvas + grain, 0, 255).astype(np.uint8), polarity

    tier_amp = (0.45 if tier == "light_texture" else 1.0) * amp_scale

    # Low-frequency illumination gradient (nonperiodic)
    gx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    gy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    ang = rng.uniform(0.0, 2.0 * np.pi)
    grad = (np.cos(ang) * gx + np.sin(ang) * gy) * rng.uniform(5.0, 13.0) * tier_amp
    canvas += grad[..., None]

    # Family-specific procedural material layer
    if family in ("pale_wood", "dark_wood"):
        # Horizontal anisotropic wood grain streaks + soft knots
        streak = _upsample_noise(np_rng, max(6, height // 3), max(2, width // 48), height, width)
        fine_streak = _upsample_noise(np_rng, max(12, height // 2), max(4, width // 24), height, width)
        wood = (streak * 9.0 + fine_streak * 4.0) * tier_amp
        canvas += wood[..., None] * np.array([1.05, 0.95, 0.80], dtype=np.float32).reshape(1, 1, 3)
    elif family == "warm_offwhite_paper":
        stain = _upsample_noise(np_rng, 3, max(4, width // 32), height, width) * 6.0 * tier_amp
        fibers = np_rng.normal(0.0, 2.5 * tier_amp, size=(height, width, 1)).astype(np.float32)
        canvas += stain[..., None] + fibers
    elif family == "dark_chalkboard":
        dust = _upsample_noise(np_rng, 4, max(6, width // 24), height, width) * 7.5 * tier_amp
        fine = np_rng.normal(0.0, 2.5 * tier_amp, size=(height, width, 1)).astype(np.float32)
        canvas += dust[..., None] + fine
    elif family == "soft_mottled_card":
        mottle = _upsample_noise(np_rng, 4, max(6, width // 28), height, width) * 9.0 * tier_amp
        fine = np_rng.normal(0.0, 2.0 * tier_amp, size=(height, width, 1)).astype(np.float32)
        canvas += mottle[..., None] + fine
    else:  # photographic_gradient
        shadow = _upsample_noise(np_rng, 3, max(4, width // 40), height, width) * 12.0 * tier_amp
        fine = np_rng.normal(0.0, 2.2 * tier_amp, size=(height, width, 1)).astype(np.float32)
        canvas += shadow[..., None] + fine

    return np.clip(canvas, 0, 255).astype(np.uint8), polarity


def choose_ink_and_audit(
    bg_roi: np.ndarray,
    alpha_64: np.ndarray,
    polarity: str,
    min_median_cr: float,
    rng: random.Random,
) -> tuple[tuple[int, int, int], float, float, float]:
    """Select plausible ink/chalk color after inspecting background under glyph core.

    Returns (ink_rgb, median_cr, p05_cr, frac_below_3).
    """
    core_mask = alpha_64 >= 180
    if int(core_mask.sum()) < 8:
        core_mask = alpha_64 >= 96
    if int(core_mask.sum()) < 4:
        core_mask = alpha_64 > 16

    bg_core = bg_roi[core_mask]
    palette = list(DARK_INKS if polarity == "light" else LIGHT_INKS)
    # Shuffle muted tones first, keep highest-contrast ink/chalk as fallback
    first = palette[0]
    rest = palette[1:]
    rng.shuffle(rest)
    candidates = rest + [first]
    if polarity == "light":
        candidates.append((8, 8, 10))
    else:
        candidates.append((250, 250, 248))

    best = None
    for ink in candidates:
        fg_arr = np.broadcast_to(np.array(ink, dtype=np.uint8).reshape(1, 3), bg_core.shape)
        crs = contrast_ratio_map(fg_arr, bg_core)
        med = float(np.median(crs))
        p05 = float(np.percentile(crs, 5))
        below3 = float((crs < 3.0).mean())
        score = (p05 >= 3.0 and med >= min_median_cr and below3 <= 0.05, med, p05)
        if best is None or score > best[0]:
            best = (score, (ink, med, p05, below3))
        if p05 >= 3.0 and med >= min_median_cr and below3 <= 0.05:
            return ink, med, p05, below3
    assert best is not None
    return best[1]
