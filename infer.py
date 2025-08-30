#!/usr/bin/env python3
"""
Usage (single image):
    python infer.py --model path/to/model.keras \
                      --input path/to/image.jpg \
                      --output outputs/mask.png

Usage (folder):
    python infer.py --model path/to/model.keras \
                      --input path/to/images_dir \
                      --output outputs/

Notes:
- The --input-size MUST match what you used during training (height, width).
- Output masks are binary PNGs (0 or 255), resized back to each image's original size.
- Optional TTA averages prediction with a horizontally flipped version.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

def parse_args():
    p = argparse.ArgumentParser(description="Run segmentation inference to produce binary masks.")
    p.add_argument("--model", required=True, type=str, help="Path to saved Keras model (.keras or .h5).")
    p.add_argument("--input", required=True, type=str, help="Input image file or directory.")
    p.add_argument("--output", required=True, type=str, help="Output file (for single image) or directory (for folder).")
    p.add_argument("--input-size", nargs=2, type=int, default=[512, 512],
                   help="Model input size as: H W (default: 512 512). Must match training.")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarizing mask (default: 0.5).")
    p.add_argument("--tta", action="store_true", help="Enable simple TTA (horizontal flip averaging).")
    p.add_argument("--exts", nargs="+", default=[".jpg", ".jpeg", ".png", ".bmp"],
                   help="Valid image extensions when --input is a folder.")
    return p.parse_args()

def load_image(path: Path):
    img = Image.open(path).convert("RGB")
    return img

def preprocess(img_pil: Image.Image, target_hw):
    """Resize to model input size and scale to [0,1]. Returns np array [H,W,3]."""
    h, w = target_hw
    img_resized = img_pil.resize((w, h), resample=Image.BILINEAR)
    arr = np.asarray(img_resized, dtype=np.float32) / 255.0
    return arr

def predict_mask(model, arr_hw3: np.ndarray, tta: bool = False):
    """arr_hw3 in [0,1], shape [H,W,3]. Returns prob map [H,W] in [0,1]."""
    x = arr_hw3[None, ...]  # [1,H,W,3]
    pred1 = model.predict(x, verbose=0)[0]  # could be [H,W,1] or [H,W]
    if pred1.ndim == 3:
        pred1 = pred1[..., 0]
    if tta:
        x_flip = x[:, :, ::-1, :]
        pred2 = model.predict(x_flip, verbose=0)[0]
        if pred2.ndim == 3:
            pred2 = pred2[..., 0]
        pred2 = pred2[:, ::-1]  # unflip
        prob = 0.5 * (pred1 + pred2)
    else:
        prob = pred1
    prob = np.clip(prob, 0.0, 1.0)
    return prob

def postprocess_to_binary(prob_hw: np.ndarray, orig_size, threshold: float):
    """Resize prob map back to original (W,H), then threshold to 0/255."""
    # PIL resize expects (W,H)
    prob_img = Image.fromarray((prob_hw * 255).astype(np.uint8))
    prob_resized = prob_img.resize((orig_size[0], orig_size[1]), resample=Image.BILINEAR)
    prob_resized_np = np.asarray(prob_resized, dtype=np.float32) / 255.0
    binary = (prob_resized_np >= threshold).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")

def ensure_out_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def infer_single(model, in_path: Path, out_path: Path, input_size, threshold: float, tta: bool):
    img = load_image(in_path)
    orig_w, orig_h = img.size
    arr = preprocess(img, (input_size[0], input_size[1]))  # (H,W)
    prob = predict_mask(model, arr, tta=tta)
    mask_img = postprocess_to_binary(prob, (orig_w, orig_h), threshold)
    # Ensure parent exists
    if out_path.parent:
        ensure_out_dir(out_path.parent)
    mask_img.save(out_path)
    print(f"[OK] Saved: {out_path}")

def collect_images(dir_path: Path, exts):
    exts_lower = set(e.lower() for e in exts)
    for p in dir_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts_lower:
            yield p

def main():
    args
