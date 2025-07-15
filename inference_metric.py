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
NUM_THREADS = int(os.getenv("NUM_THREADS", "4"))

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
    if not info:
        vec = np.zeros(6, np.float32)
    else:
        if isinstance(info, list):
            info = info[0] if info else {}
        vec = np.asarray([float(info.get(k, 0.0)) for k in KEYS], np.float32)
    vec = np.clip(vec, MINS, MAXS)
    vec = (vec - MINS) * 255.0 / (MAXS - MINS)
    return vec.astype(np.float32)

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

    n_classes = max(cat["id"] for cat in anno_raw["categories"]) + 1
    cm = np.zeros((n_classes, n_classes), np.int64)

    interp = load_interpreter(MODEL_PATH)
    input_details = interp.get_input_details()
    output_detail = interp.get_output_details()[0]

    img_in = next(d for d in input_details if len(d["shape"]) == 4)
    sensor_in = next(d for d in input_details if d is not img_in)
    target_hw: Tuple[int, int] = (img_in["shape"][2], img_in["shape"][1])

    # Evaluation loop
    for img_id, img_info in tqdm(id2info.items(), desc="Evaluating"):
        img_path = IMAGE_DIR / img_info["file_name"]
        if not img_path.exists():
            continue
        # Load image
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, target_hw, interpolation=cv2.INTER_LINEAR)
        img_float = img_rgb.astype(np.float32) / 255.0  # 0~1

        # Build GT mask
        anns = imgid_to_ann.get(img_id, [])
        mask = np.zeros(target_hw[::-1], np.uint8)
        for ann in anns:
            cid = ann["category_id"]
            for poly in ann["segmentation"]:
                pts = np.asarray(poly, np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], cid)

        # Sensor vector
        sensor_vec = sensor_to_vec(img_info.get("sensor_info"))

        # Prepare tensors
        img_uint8 = (img_float * 255.0).round().astype(np.uint8)[None, ...]
        sensor_uint8 = np.round(sensor_vec).astype(np.uint8)[None, ...]
        interp.set_tensor(img_in["index"], img_uint8)
        interp.set_tensor(sensor_in["index"], sensor_uint8)
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