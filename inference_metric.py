#!/usr/bin/env python
"""Standalone evaluation script (no external project imports).

• 입력 COCO JSON : /workspace/data/COCO/test_without_street.json
• 이미지 폴더     : /workspace/data/images/
• 모델            : seg_model_sensor_int8.tflite (모델 루트에 존재한다고 가정)
• 추론            : 샘플당 1장, tqdm 진행 바 표시
• 출력            : Pixel Acc, mIoU, mDice, FW IoU
"""

from __future__ import annotations

import json, os, argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm

# tflite runtime 우선
try:
    import tflite_runtime.interpreter as tflite  # type: ignore
except ModuleNotFoundError:
    tflite = tf.lite  # type: ignore

# --------------------------------------------------------------------------------------
# 고정 경로 설정 (필요 시 수정)
# --------------------------------------------------------------------------------------
DATA_ROOT = Path("/workspace/data")
IMAGE_DIR = DATA_ROOT / "images"
ANNO_PATH = DATA_ROOT / "COCO" / "test_without_street.json"
MODEL_PATH = Path("seg_model_sensor_int8.tflite")
NUM_THREADS = int(os.getenv("NUM_THREADS", "16"))

# --------------------------------------------------------------------------------------
# Sensor helper (0~255 scaling)
# --------------------------------------------------------------------------------------


MINS = np.asarray([-100, 0, 950, -90, -180, 0], np.float32)
MAXS = np.asarray([100, 100, 1050, 90, 180, 1000], np.float32)

KEYS = [
    "objectTemp",
    "humi",
    "pressure",
    "latitude",
    "longitude",
    "height",
]


def sensor_to_vec(info: Dict[str, Any] | None) -> np.ndarray:
    """Convert sensor dict/list into 0 – 255 float32 vector (length=6)."""

    if not info:
        vec = np.zeros(6, np.float32)
    else:
        if isinstance(info, list):
            info = info[0] if info else {}
        vec = np.asarray([float(info.get(k, 0.0)) for k in KEYS], np.float32)

    vec = np.clip(vec, MINS, MAXS)
    vec = (vec - MINS) * 255.0 / (MAXS - MINS)
    return vec.astype(np.float32)

# -----------------------------------------------------------------------------
# Polygon → mask rasterizer (stand-alone copy from ImgClsDatasetTF)
# -----------------------------------------------------------------------------


def get_mask_from_polygon(img_shape: Tuple[int, int], num_categories: int, annotations: list[Dict]) -> np.ndarray:  # type: ignore[valid-type]
    """Rasterise polygon annotations to an H×W integer mask.

    Each *annotation* must contain ``category_id`` and ``segmentation`` (list of
    polygons).  Pixels hold the category id (0 … N-1). Background == 0.
    """

    height, width = img_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    for ann in annotations:
        cid = ann.get("category_id", 0)
        segm = ann.get("segmentation", [])
        if not isinstance(segm, list):
            # Unsupported RLE segmentation – skip
            continue
        for poly in segm:
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], cid)
    return mask.astype(np.uint8)

# --------------------------------------------------------------------------------------
# Metrics helpers
# --------------------------------------------------------------------------------------

def update_cm(cm: np.ndarray, gt: np.ndarray, pred: np.ndarray):
    cm += tf.math.confusion_matrix(gt.flatten(), pred.flatten(), num_classes=cm.shape[0]).numpy()


def compute_metrics(cm: np.ndarray):
    diag = np.diag(cm).astype(np.float64)
    rows = cm.sum(axis=1).astype(np.float64)
    cols = cm.sum(axis=0).astype(np.float64)
    total = cm.sum().astype(np.float64)

    pixel_acc = diag.sum() / total if total else 0
    union = rows + cols - diag
    iou = np.where(union > 0, diag / union, 0.0)
    mean_iou = iou.mean()
    dice = np.where(rows + cols > 0, 2 * diag / (rows + cols), 0.0)
    mean_dice = dice.mean()
    freq = rows / total if total else np.zeros_like(rows)
    fw_iou = (freq * iou).sum()
    return pixel_acc, mean_iou, mean_dice, fw_iou, iou, dice

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def load_interpreter(model: Path):
    interp = tflite.Interpreter(model_path=str(model), num_threads=NUM_THREADS)
    interp.allocate_tensors()
    return interp


def main():
    # Load annotations
    with open(ANNO_PATH) as f:
        anno_raw = json.load(f)

    # Map image id -> annotations indices, and id -> info
    id2info = {img["id"]: img for img in anno_raw["images"]}
    imgid_to_ann = {}
    for ann in anno_raw["annotations"]:
        imgid_to_ann.setdefault(ann["image_id"], []).append(ann)

    # Only evaluate images that have annotations (align with ImgClsDatasetTF)
    eval_ids = [i for i in id2info.keys() if i in imgid_to_ann]

    n_classes = max(cat["id"] for cat in anno_raw["categories"]) + 1
    cm = np.zeros((n_classes, n_classes), np.int64)

    interp = load_interpreter(MODEL_PATH)
    input_details = interp.get_input_details()
    output_detail = interp.get_output_details()[0]

    img_in = next(d for d in input_details if len(d["shape"]) == 4)
    sensor_in = next(d for d in input_details if d is not img_in)
    target_hw: Tuple[int, int] = (img_in["shape"][2], img_in["shape"][1])

    # Evaluation loop
    for img_id in tqdm(eval_ids, desc="Evaluating"):
        img_info = id2info[img_id]
        img_path = IMAGE_DIR / img_info["file_name"]
        if not img_path.exists():
            continue
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img_rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb_orig.shape[:2]  # 저장해 두기 (폴리곤 좌표계)

        # Build GT mask (original resolution → input size)
        anns = imgid_to_ann[img_id]
        mask_full = get_mask_from_polygon((orig_h, orig_w), n_classes, anns)
        mask_tf = tf.convert_to_tensor(mask_full[..., None])  # add channel dim
        mask_tf = tf.image.resize(mask_tf, target_hw[::-1], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask = tf.squeeze(mask_tf, -1).numpy().astype(np.uint8)

        # Resize image for model using Bilinear (align with tf pipeline)
        img_tf = tf.convert_to_tensor(img_rgb_orig, dtype=tf.float32)  # H W 3, 0-255
        img_tf = img_tf / 255.0  # 0-1 float32
        img_tf = tf.image.resize(img_tf, target_hw[::-1], method=tf.image.ResizeMethod.BILINEAR)
        img_float = img_tf.numpy()

        # Sensor vector
        sensor_vec = sensor_to_vec(img_info.get("sensor_info"))

        # Prepare tensors
        if img_in["dtype"] == np.uint8:
            img_tensor = (img_float * 255.0).round().astype(np.uint8)[None, ...]
        else:  # float32
            img_tensor = img_float[None, ...].astype(np.float32)

        if sensor_in["dtype"] == np.uint8:
            sensor_tensor = np.round(sensor_vec).astype(np.uint8)[None, ...]
        else:  # float32
            sensor_tensor = sensor_vec[None, ...].astype(np.float32)

        interp.set_tensor(img_in["index"], img_tensor)
        interp.set_tensor(sensor_in["index"], sensor_tensor)
        interp.invoke()
        pred = interp.get_tensor(output_detail["index"])[0]
        pred_cls = np.argmax(pred, axis=-1).astype(np.int32)

        update_cm(cm, mask, pred_cls)

    # Metrics
    pixel_acc, mean_iou, mean_dice, fw_iou, iou_per, dice_per = compute_metrics(cm)
    print("\n=== 평가 결과 (INT8) ===")
    for cid in range(n_classes):
        print(f"Class {cid}: IoU={iou_per[cid]:.4f}, Dice={dice_per[cid]:.4f}")
    print(f"Pixel Accuracy         : {pixel_acc:.4f}")
    print(f"Mean IoU               : {mean_iou:.4f}")
    print(f"Mean Dice Coefficient  : {mean_dice:.4f}")
    print(f"Frequency Weighted IoU : {fw_iou:.4f}")


if __name__ == "__main__":
    main() 