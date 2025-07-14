#!/usr/bin/env python
"""Batch evaluation script for INT8 TFLite Coral-DeepLab model (image + sensor).

Usage
-----
$ python test_with_int8.py \
    --tflite seg_model_sensor_int8.tflite \
    --delegate edgetpu  # (옵션)

테스트 split( config/model_cfg.py 에 정의 ) 전체에 대해 Pixel Accuracy,
Mean IoU, Mean Dice coefficient, Frequency-Weighted IoU 를 계산한다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import model_cfg as cfg_mod
from coral_deeplab.utils.img_cls_dataset import ImgClsDatasetTF

# tflite-runtime 우선, 없으면 tensorflow.lite 로 대체
try:
    import tflite_runtime.interpreter as tflite  # type: ignore
except ModuleNotFoundError:  # fallback to full TF
    tflite = tf.lite  # type: ignore

################################################################################
# Delegate & Interpreter helpers
################################################################################

def _load_delegate(name: str | None):
    """HW 가속 delegate 로드 (현재 EdgeTPU 만 지원)."""

    if name is None:
        return None
    name = name.lower()
    if name == "edgetpu":
        try:
            return tflite.load_delegate("libedgetpu.so.1")
        except ValueError:
            print("[warn] EdgeTPU delegate 로드 실패 – CPU(XNNPACK) 로 진행합니다")
            return None
    print(f"[warn] 알 수 없는 delegate: {name} – delegate 없이 진행")
    return None


def _new_interpreter(model_path: Path, delegate_name: str | None = None):
    delegate = _load_delegate(delegate_name)
    kwargs: Dict[str, Any] = {"model_path": str(model_path), "num_threads": int(os.getenv("NUM_THREADS", "4"))}
    if delegate is not None:
        kwargs["experimental_delegates"] = [delegate]
    return tflite.Interpreter(**kwargs)

################################################################################
# Metric helpers
################################################################################

def update_confusion_matrix(conf: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
    """누적 confusion matrix 갱신"""

    cm = tf.math.confusion_matrix(
        y_true.flatten(), y_pred.flatten(), num_classes=conf.shape[0]
    ).numpy()
    conf += cm


def compute_metrics(conf: np.ndarray):
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
# Argument parsing
################################################################################

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate INT8 TFLite model on test set")
    p.add_argument("--tflite", default="seg_model_sensor_int8.tflite", help="INT8 model path")
    p.add_argument("--delegate", default=None, help="Delegate name (e.g., edgetpu)")
    p.add_argument("--input_size", type=int, default=513, help="Square input resolution")
    p.add_argument("--n_classes", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=1, help="Batch size (fixed to 1 for TFLite, keep=1)")
    return p.parse_args()

################################################################################
# Main evaluation logic
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
    # Build tf.data.Dataset (batch_size=1 → sample 단위)
    # --------------------------------------------------------------
    ds_gen = ImgClsDatasetTF(data_cfg, test_raw, transforms=None)
    ds = ds_gen.as_dataset(batch_size=1, shuffle=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # --------------------------------------------------------------
    # Prepare TFLite interpreter
    # --------------------------------------------------------------
    interpreter = _new_interpreter(Path(args.tflite), args.delegate)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 이미지 입력(4-D) vs 센서 입력(<=2-D) 자동 판별
    img_in = next((d for d in input_details if len(d["shape"]) == 4), input_details[0])
    sensor_in = next((d for d in input_details if d is not img_in), None)
    if sensor_in is None:
        raise RuntimeError("센서 입력을 찾을 수 없습니다 – 모델이 센서 분기를 포함하고 있나요?")

    target_hw: Tuple[int, int] = (img_in["shape"][2], img_in["shape"][1])  # (W, H)

    # --------------------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------------------
    conf_mat = np.zeros((n_classes, n_classes), dtype=np.int64)

    for img_tf, mask_tf, sensor_tf, _ in tqdm(ds, desc="Evaluating"):
        # Tensor → ndarray 변환
        img_np: np.ndarray = img_tf.numpy()[0]  # (H, W, 3), float32 0~1
        mask_np: np.ndarray = mask_tf.numpy()[0]  # (H, W), int32
        sensor_np: np.ndarray = sensor_tf.numpy()[0]  # (6,), float32 0~255

        # Resize (bilinear / nearest) to model input resolution
        img_res = tf.image.resize(img_np, target_hw[::-1], method=tf.image.ResizeMethod.BILINEAR).numpy()
        mask_res = tf.image.resize(tf.expand_dims(mask_np, -1), target_hw[::-1], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR).numpy()
        mask_res = mask_res.squeeze(-1).astype(np.int32)

        # ------------------------------------------------------------------
        # 입력 dtype 에 따라 스케일/캐스팅
        # ------------------------------------------------------------------
        def _prepare(arr: np.ndarray, info: Dict[str, Any]):
            """이미지/센서 배열을 info["dtype"] 에 맞춰 변환"""
            dtype = info["dtype"]
            if dtype == np.uint8:
                if arr.dtype != np.uint8:
                    # 이미지(0~1) vs 센서(0~255) 구분: max 값으로 heuristic
                    if arr.max() <= 1.0:
                        arr = (arr * 255.0).round()
                    # 이미 0~255 실수 범위일 경우 그대로 라운딩
                    arr = np.clip(arr, 0, 255).round().astype(np.uint8)
            elif dtype == np.float32:
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
            else:
                raise ValueError(f"지원되지 않는 입력 dtype: {dtype}")
            return np.expand_dims(arr, 0)

        img_in_arr = _prepare(img_res, img_in)
        sensor_in_arr = _prepare(sensor_np, sensor_in)

        # Set tensors
        interpreter.set_tensor(img_in["index"], img_in_arr)
        interpreter.set_tensor(sensor_in["index"], sensor_in_arr)
        interpreter.invoke()

        # Get prediction
        pred_logits = interpreter.get_tensor(output_details[0]["index"])
        pred_cls = np.argmax(pred_logits[0], axis=-1).astype(np.int32)

        # Update metrics
        update_confusion_matrix(conf_mat, mask_res, pred_cls)

    # --------------------------------------------------------------
    # Compute & display results
    # --------------------------------------------------------------
    pixel_acc, mean_iou, mean_dice, freq_weighted_iou, iou_per_cls, dice_per_cls = compute_metrics(conf_mat)

    print("\n=== 평가 지표 ===")
    for cid in range(n_classes):
        print(f"Class {cid}: IoU={iou_per_cls[cid]:.4f}, Dice={dice_per_cls[cid]:.4f}")
    print(f"Pixel Accuracy: {pixel_acc:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Dice Coefficient: {mean_dice:.4f}")
    print(f"Frequency Weighted IoU: {freq_weighted_iou:.4f}")


if __name__ == "__main__":
    main() 