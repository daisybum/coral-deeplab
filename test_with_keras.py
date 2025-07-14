#!/usr/bin/env python
"""Batch evaluation script for Coral-DeepLab Keras checkpoints.

Usage example
-------------
$ python test_with_keras.py \
    --ckpt checkpoints_tf/epoch_195.keras \
    --batch_size 32 --model deeplabv3plus

The script loads the *test* split defined in ``config/model_cfg.py`` and
computes mean IoU and mean Dice across the entire dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tqdm import tqdm

from config import model_cfg as cfg_mod
from coral_deeplab.utils.img_cls_dataset import ImgClsDatasetTF
from coral_deeplab.utils.transforms import get_transforms  # not used but kept for future
from coral_deeplab.applications import CoralDeepLabV3, CoralDeepLabV3Plus

################################################################################
# Helper functions
################################################################################

def build_model(name: str, input_shape=(513, 513, 3), n_classes: int = 6):
    if name == "deeplabv3":
        return CoralDeepLabV3(input_shape=input_shape, n_classes=n_classes)
    elif name == "deeplabv3plus":
        return CoralDeepLabV3Plus(input_shape=input_shape, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model {name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CoralDeepLab Keras checkpoint on test set")
    p.add_argument("--ckpt", required=True, help="Keras checkpoint (.keras) path")
    p.add_argument("--model", default="deeplabv3plus", choices=["deeplabv3", "deeplabv3plus"], help="Model type")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--input_size", type=int, default=513)
    p.add_argument("--n_classes", type=int, default=6)
    return p.parse_args()


################################################################################
# Metric helpers
################################################################################

def update_confusion_matrix(conf: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
    """Accumulate confusion matrix given ground-truth and prediction."""
    num_classes = conf.shape[0]
    cm = tf.math.confusion_matrix(
        y_true.flatten(), y_pred.flatten(), num_classes=num_classes
    ).numpy()
    conf += cm


def compute_iou_dice(conf: np.ndarray):
    diag = np.diag(conf).astype(np.float64)
    rows = conf.sum(axis=1).astype(np.float64)
    cols = conf.sum(axis=0).astype(np.float64)
    union = rows + cols - diag
    iou = np.where(union > 0, diag / union, 0.0)
    dice = np.where(rows + cols > 0, 2 * diag / (rows + cols), 0.0)
    return iou, dice

# -----------------------------------------------------------------------------
# 추가 메트릭 (Pixel Accuracy, Frequency-Weighted IoU)
# -----------------------------------------------------------------------------


def compute_metrics(conf: np.ndarray):
    """confusion matrix 기반 종합 메트릭 계산."""

    diag = np.diag(conf).astype(np.float64)
    rows = conf.sum(axis=1).astype(np.float64)
    cols = conf.sum(axis=0).astype(np.float64)
    total = conf.sum().astype(np.float64)

    pixel_acc = diag.sum() / total if total > 0 else 0.0

    union = rows + cols - diag
    iou = np.where(union > 0, diag / union, 0.0)
    mean_iou = iou.mean()

    dice = np.where(rows + cols > 0, 2 * diag / (rows + cols), 0.0)
    mean_dice = dice.mean()

    freq = rows / total if total > 0 else np.zeros_like(rows)
    freq_weighted_iou = (freq * iou).sum()

    return pixel_acc, mean_iou, mean_dice, freq_weighted_iou, iou, dice


################################################################################
# Main
################################################################################

def main():
    args = parse_args()

    # --------------------------------------------------------------
    # Load test annotations
    # --------------------------------------------------------------
    data_cfg = cfg_mod.DATA_CFG
    ann_path = os.path.join(data_cfg.root_path, data_cfg.test_anno_path)
    with open(ann_path, "r") as f:
        test_raw: Dict[str, Any] = json.load(f)

    dataset_n_classes = max(cat["id"] for cat in test_raw["categories"]) + 1
    n_classes = args.n_classes or dataset_n_classes

    # --------------------------------------------------------------
    # Build dataset
    # --------------------------------------------------------------
    ds_gen = ImgClsDatasetTF(data_cfg, test_raw, transforms=None)
    ds = ds_gen.as_dataset(batch_size=args.batch_size, shuffle=False)

    TARGET_HW = (args.input_size, args.input_size)

    def _resize_img(img):
        return tf.image.resize(img, TARGET_HW, method=tf.image.ResizeMethod.BILINEAR)

    def _resize_mask(mask):
        mask = tf.expand_dims(mask, -1)
        mask = tf.image.resize(mask, TARGET_HW, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.squeeze(mask, -1)

    ds = ds.map(
        lambda img, mask, sensors, ann: (
            (
                _resize_img(img),
                sensors,  # already 0-255 float32 from _sensor_to_vec
            ),
            tf.cast(_resize_mask(mask), tf.int32),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Prefetch for speed
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # --------------------------------------------------------------
    # Build & load model
    # --------------------------------------------------------------
    input_shape = (args.input_size, args.input_size, 3)
    model = build_model(args.model, input_shape=input_shape, n_classes=n_classes)
    model.load_weights(args.ckpt)

    # --------------------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------------------
    conf_mat = np.zeros((n_classes, n_classes), dtype=np.int64)

    for (imgs, sensors), masks in tqdm(ds, desc="Batch eval"):
        preds = model.predict([imgs, sensors], verbose=0)
        preds = np.argmax(preds, axis=-1).astype(np.int32)
        masks_np = masks.numpy().astype(np.int32)
        for pr, gt in zip(preds, masks_np):
            update_confusion_matrix(conf_mat, gt, pr)

    pixel_acc, mean_iou, mean_dice, freq_weighted_iou, iou, dice = compute_metrics(conf_mat)

    print("\n=== 평가 지표 ===")
    for cid in range(n_classes):
        print(f"Class {cid}: IoU={iou[cid]:.4f}, Dice={dice[cid]:.4f}")
    print(f"Pixel Accuracy: {pixel_acc:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Dice Coefficient: {mean_dice:.4f}")
    print(f"Frequency Weighted IoU: {freq_weighted_iou:.4f}")


if __name__ == "__main__":
    main() 