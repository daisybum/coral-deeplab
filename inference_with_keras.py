"""inference_with_keras.py
===========================
TensorFlow/Keras 체크포인트(.keras) 가중치를 로드하여 이미지를 세그멘테이션하는 스크립트.

사용 예)
---------
$ python inference_with_keras.py \
    --ckpt checkpoints_tf/epoch_110.keras \
    --input example/20220428_000_40P0S1R1AX_0_20220702_065412.jpg \
    --save_mask --output_dir results

이미지 디렉터리를 주면 내부의 *.jpg|png|bmp* 파일을 일괄 처리합니다.
센서 입력이 없으면 0 벡터가 사용됩니다.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import cv2  # type: ignore
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from PIL import Image  # type: ignore

# 내부 모듈 – CoralDeepLab 모델 정의
from coral_deeplab.applications import CoralDeepLabV3, CoralDeepLabV3Plus

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("inference_keras")


# --------------------------------------------------------------------------------------
# Sensor helpers
# --------------------------------------------------------------------------------------

def _sensor_to_vec(sensor_data: dict | list | None) -> np.ndarray:
    """annotation JSON의 sensor_info → (1, 6) float32 벡터 변환"""

    if not sensor_data:
        return np.zeros((1, 6), dtype=np.float32)
    if isinstance(sensor_data, list):
        sensor_data = sensor_data[0] if sensor_data else {}

    keys = [
        "objectTemp",
        "humi",
        "pressure",
        "latitude",
        "longitude",
        "height",
    ]
    vec = [float(sensor_data.get(k, 0.0)) for k in keys]
    return np.asarray([vec], dtype=np.float32)


def _parse_sensor_values_manual(s: str | None) -> np.ndarray | None:
    if s is None:
        return None
    vals = [float(v.strip()) for v in s.split(",")]
    return np.asarray([vals], dtype=np.float32)


# --------------------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------------------


def _prepare_input(pil: Image.Image, target_hw: Tuple[int, int]) -> np.ndarray:
    """PIL 이미지를 (1, H, W, C) float32 0~1 범위로 변환."""

    w, h = target_hw
    img = pil.resize((w, h), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def _overlay_mask_on_image(
    pil_img: Image.Image,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> Image.Image:
    """세그멘테이션 마스크를 원본 위에 overlay 및 반환."""

    mask_img = Image.fromarray(mask.astype(np.uint8))
    mask_resized = mask_img.resize(pil_img.size, Image.NEAREST)

    pil_arr = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
    mask_arr = np.asarray(mask_resized, dtype=np.uint8)

    m = mask_arr > 0
    if m.ndim == 3:
        m = m[..., 0]

    overlay_arr = pil_arr.copy()
    overlay_color = np.array(color, dtype=np.uint8)
    overlay_arr[m] = (
        pil_arr[m].astype(np.float32) * (1.0 - alpha) + overlay_color.astype(np.float32) * alpha
    ).astype(np.uint8)

    return Image.fromarray(overlay_arr)


# --------------------------------------------------------------------------------------
# Inference core
# --------------------------------------------------------------------------------------

def _build_model(model_name: str, input_shape=(513, 513, 3), n_classes: int = 6):
    if model_name == "deeplabv3":
        return CoralDeepLabV3(input_shape=input_shape, n_classes=n_classes)
    elif model_name == "deeplabv3plus":
        return CoralDeepLabV3Plus(input_shape=input_shape, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model name {model_name}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CoralDeepLab Keras 추론 스크립트")
    p.add_argument("--input", required=True, help="입력 이미지 파일 또는 디렉터리")
    p.add_argument("--ckpt", required=True, help="Keras 체크포인트(.keras) 가중치 경로")
    p.add_argument("--model", default="deeplabv3plus", choices=["deeplabv3", "deeplabv3plus"], help="모델 종류")
    p.add_argument("--input_size", type=int, default=513, help="이미지 리사이즈 크기(H=W)")
    p.add_argument("--n_classes", type=int, default=6, help="클래스 수")
    sensor_grp = p.add_mutually_exclusive_group()
    sensor_grp.add_argument("--sensor_json", default=None, help="단일 센서 JSON 파일(모든 이미지에 공통 적용)")
    sensor_grp.add_argument("--sensor_dir", default=None, help="이미지와 동일한 이름의 .json 파일이 위치한 폴더")
    p.add_argument("--sensor_values", default=None, help="쉼표 구분 수동 센서 값(ex: 24.1,70,1013,37.12,126.9,2.5)")
    p.add_argument("--output_dir", default="inference_output", help="결과 저장 폴더")
    p.add_argument("--save_mask", action="store_true", help="세그멘테이션 마스크/overlay 저장 여부")
    p.add_argument("--mask_size", type=int, default=513, help="리사이즈할 출력 마스크 한 변의 크기 (정사각형)")
    return p


def main():
    args = _build_argparser().parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 센서 입력 로딩(전역 또는 per-image)
    # ------------------------------------------------------------------

    global_sensor_arr: np.ndarray | None = None

    if args.sensor_dir:
        logger.info("Per-image sensor JSON 폴더 사용 → %s", args.sensor_dir)
    elif args.sensor_json:
        logger.info("Loading sensor JSON → %s", args.sensor_json)
        with open(args.sensor_json, "r") as f:
            sensor_raw = json.load(f)
        global_sensor_arr = _sensor_to_vec(sensor_raw)
    else:
        global_sensor_arr = _parse_sensor_values_manual(args.sensor_values) or np.zeros((1, 6), dtype=np.float32)

    # ------------------------------------------------------------------
    # 모델 로드
    # ------------------------------------------------------------------
    input_shape = (args.input_size, args.input_size, 3)
    logger.info("Building model %s (input %dx%d) …", args.model, args.input_size, args.input_size)

    def _try_load(model_name: str):
        m = _build_model(model_name, input_shape=input_shape, n_classes=args.n_classes)
        m.load_weights(args.ckpt)
        return m

    try:
        model = _try_load(args.model)
    except ValueError as e:
        # 가중치 레이어 수 불일치 등 – 다른 모델 시도 (v3 <-> v3plus)
        alt_model = "deeplabv3plus" if args.model == "deeplabv3" else "deeplabv3"
        logger.warning("%s 모델로 weight 로드 실패: %s", args.model, e)
        logger.info("대안 모델 %s 로 재시도합니다…", alt_model)
        try:
            model = _try_load(alt_model)
            logger.info("Checkpoint loaded with 모델 %s", alt_model)
        except Exception as e2:
            logger.error("두 모델 모두 로드 실패 – --model 옵션 확인 필요: %s", e2)
            raise
    else:
        logger.info("Checkpoint loaded → %s", args.ckpt)

    # ------------------------------------------------------------------
    # 입력 이미지 리스트 확보
    # ------------------------------------------------------------------
    in_paths: list[Path]
    p = Path(args.input)
    if p.is_dir():
        in_paths = sorted([f for f in p.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    else:
        in_paths = [p]

    # ------------------------------------------------------------------
    # 추론 루프
    # ------------------------------------------------------------------
    for img_path in in_paths:
        ts0 = time.time()
        # 이미지 열기
        with Image.open(img_path) as pil:
            pil = pil.convert("RGB")
            inp_arr = _prepare_input(pil, (args.input_size, args.input_size))

        # 이미지별 센서 벡터 결정
        if args.sensor_dir:
            json_path = Path(args.sensor_dir) / f"{img_path.stem}.json"
            if json_path.exists():
                with open(json_path, "r") as f:
                    sensor_raw = json.load(f)
                sensor_arr_use = _sensor_to_vec(sensor_raw)
            else:
                sensor_arr_use = np.zeros((1, 6), dtype=np.float32)
        else:
            sensor_arr_use = global_sensor_arr

        # 모델 예측
        preds = model.predict([inp_arr, sensor_arr_use], verbose=0)[0]  # (H, W, C)

        # 후처리: argmax → (H, W)
        mask = np.argmax(preds, axis=-1).astype(np.uint8)

        elapsed = time.time() - ts0
        logger.info("Processed %s – %.3f s", img_path.name, elapsed)

        if args.save_mask:
            if mask.shape[0] != args.mask_size:
                mask = cv2.resize(mask, (args.mask_size, args.mask_size), interpolation=cv2.INTER_NEAREST)
            mask_img = Image.fromarray(mask)
            mask_img.save(out_dir / f"{img_path.stem}_mask.png")

            # Overlay 저장
            with Image.open(img_path) as orig_pil:
                overlay = _overlay_mask_on_image(orig_pil, mask)
                overlay.save(out_dir / f"{img_path.stem}_overlay.png")


if __name__ == "__main__":
    main() 