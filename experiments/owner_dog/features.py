"""Fixed classical image descriptors for the new 200-set experiment.

These calculations contain no learned extractor, identity labels, PCA or model
selection. Images are composited on white before appearance descriptors are
computed; HSV counts are additionally weighted by the supplied alpha channel.
"""

from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps
from scipy.signal import fftconvolve
from skimage.color import rgb2hsv
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor_kernel


FEATURE_DIMENSIONS = {
    "hog_lbp": 1924,
    "hog_only": 1764,
    "lbp_only": 160,
    "enhanced": 2724,
    "gabor_lbp": 1120,
}
LBP_SETTINGS = ((8, 1), (16, 2), (24, 3))
GABOR_FREQUENCIES = (0.15, 0.30)
GABOR_ORIENTATIONS = tuple(np.arange(4) * np.pi / 4)


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    # Match the project's row-wise reduction exactly, including floating rounding.
    norm = np.linalg.norm(vector[None, :], axis=1, keepdims=True)[0, 0]
    return vector / max(float(norm), 1e-12)


def _combine(*blocks):
    return _unit(np.concatenate([_unit(block) for block in blocks]))


def composite_white(image):
    """Return RGB with no dependence on RGB bytes under fully transparent pixels."""
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, rgba).convert("RGB")


def _lbp_histogram(gray, neighbors, radius):
    lbp = local_binary_pattern(gray, neighbors, radius, method="uniform")
    bins = neighbors + 2
    cells = []
    for row in np.array_split(lbp, 4, axis=0):
        for cell in np.array_split(row, 4, axis=1):
            hist = np.bincount(cell.astype(int).ravel(), minlength=bins)[:bins].astype(float)
            cells.extend(hist / max(hist.sum(), 1))
    return np.asarray(cells, dtype=np.float64)


def _color_histogram(rgb, alpha):
    small_rgb = np.asarray(rgb.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float64) / 255
    small_alpha = np.asarray(alpha.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float64) / 255
    hsv = rgb2hsv(small_rgb)
    histograms = []
    for channel in range(3):
        counts, _ = np.histogram(hsv[:, :, channel], bins=32, range=(0, 1), weights=small_alpha)
        histograms.extend(counts / max(float(counts.sum()), 1e-12))
    return np.asarray(histograms, dtype=np.float64)


@lru_cache(maxsize=1)
def _gabor_kernels():
    return tuple(
        np.asarray(gabor_kernel(frequency=frequency, theta=float(theta)), dtype=np.complex128)
        for frequency in GABOR_FREQUENCIES
        for theta in GABOR_ORIENTATIONS
    )


def _gabor_statistics(gray):
    image = gray.astype(np.float64) / 255
    values = []
    for kernel in _gabor_kernels():
        padding = ((kernel.shape[0] // 2,) * 2, (kernel.shape[1] // 2,) * 2)
        padded = np.pad(image, padding, mode="reflect")
        magnitude = np.abs(fftconvolve(padded, kernel, mode="valid"))
        if magnitude.shape != image.shape:
            raise RuntimeError("Unexpected Gabor convolution geometry")
        for row in np.array_split(magnitude, 4, axis=0):
            for cell in np.array_split(row, 4, axis=1):
                values.extend((float(cell.mean()), float(cell.std())))
    return np.asarray(values, dtype=np.float64)


def extract_variants(image):
    """Extract the five fixed descriptors from one image, without fitting anything."""
    rgba = image.convert("RGBA")
    rgb = composite_white(rgba)
    gray = np.asarray(ImageOps.grayscale(rgb).resize((64, 64), Image.Resampling.BILINEAR))
    shape = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=False,
        feature_vector=True,
        channel_axis=None,
    )
    lbp_blocks = [_lbp_histogram(gray, p, radius) for p, radius in LBP_SETTINGS]
    multiscale_lbp = _combine(*lbp_blocks)
    color = _color_histogram(rgb, rgba.getchannel("A"))
    gabor = _gabor_statistics(gray)
    variants = {
        "hog_lbp": _combine(shape, lbp_blocks[0]),
        "hog_only": _unit(shape),
        "lbp_only": _unit(lbp_blocks[0]),
        "enhanced": _combine(shape, multiscale_lbp, color),
        "gabor_lbp": _combine(gabor, multiscale_lbp),
    }
    for name, vector in variants.items():
        if vector.shape != (FEATURE_DIMENSIONS[name],) or not np.isfinite(vector).all():
            raise RuntimeError(f"Invalid {name} descriptor")
    return variants


def describe_config():
    """Return JSON-compatible definitions, including normalisation and geometry."""
    return {
        "implementation": "features_owner_dog_200_20261101.py",
        "learned_feature_parameters": False,
        "dimensions": dict(FEATURE_DIMENSIONS),
        "appearance_preprocessing": "PIL RGBA alpha-composite on white RGB, grayscale, resize 64x64 BILINEAR",
        "hog": {
            "orientations": 9,
            "pixels_per_cell": [8, 8],
            "cells_per_block": [2, 2],
            "block_norm": "L2-Hys",
            "transform_sqrt": False,
            "dimension": 1764,
        },
        "lbp": {
            "method": "uniform",
            "spatial_grid": [4, 4],
            "per_cell_normalization": "L1 histogram; bins=P+2, including non-uniform bin",
            "baseline": {"P": 8, "R": 1, "dimension": 160},
            "multiscale": [
                {"P": p, "R": radius, "bins_per_cell": p + 2, "dimension": 16 * (p + 2)}
                for p, radius in LBP_SETTINGS
            ],
            "multiscale_dimension": 864,
            "multiscale_combination": "L2-normalize each radius vector, concatenate, L2-normalize",
        },
        "color": {
            "space": "HSV via skimage.color.rgb2hsv",
            "input_size": [128, 128],
            "rgb_input": "white-composited RGB resized BILINEAR",
            "pixel_weights": "original alpha resized BILINEAR, scaled to [0,1]; fully transparent pixels have zero weight",
            "bins_per_channel": 32,
            "histogram_range": [0, 1],
            "normalization": "per-channel L1 before block L2",
            "dimension": 96,
            "opaque_input_limit": "Without a supplied alpha mask, background pixels receive weight 1",
        },
        "gabor": {
            "frequencies": list(GABOR_FREQUENCIES),
            "orientations_radians": [float(theta) for theta in GABOR_ORIENTATIONS],
            "kernel": "skimage.filters.gabor_kernel defaults except frequency and theta",
            "convolution": "scipy.signal.fftconvolve, reflect-pad by kernel half-width, valid output",
            "response": "complex magnitude",
            "spatial_grid": [4, 4],
            "cell_statistics": ["mean", "population_std"],
            "dimension": 256,
        },
        "final_combination": "L2-normalize each named block, concatenate and L2-normalize; zero blocks stay zero",
        "variant_blocks": {
            "hog_lbp": ["hog", "baseline_lbp"],
            "hog_only": ["hog"],
            "lbp_only": ["baseline_lbp"],
            "enhanced": ["hog", "multiscale_lbp", "foreground_hsv"],
            "gabor_lbp": ["gabor", "multiscale_lbp"],
        },
        "interpretation_limit": "Fixed spatial cells are not detected eyes, ears, facial landmarks or semantic regions",
    }
