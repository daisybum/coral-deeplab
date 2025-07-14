# -*- coding: utf-8 -*-
"""inference.py
=================
Coral-DeepLab 프로젝트용 TFLite 추론 스크립트.

본 모듈은 (1) 세그멘테이션 .tflite 모델과 (2) 이미지 분류 .tflite 모델을 동시에
로드한 후, 입력 이미지에 대해 두 모델을 연속 호출하여 결과를 반환한다.

사용 예)
---------
$ python inference.py \
    --seg_model seg_model_sensor_int8.tflite \
    --cls_model cls_model_int8.tflite \
    --input examples/bird.bmp \
    --output_dir output_results

추가로 --delegate edgetpu 플래그를 주면 EdgeTPU delegate를 사용한다.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import time
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np  # type: ignore
from PIL import Image  # type: ignore

# tflite-runtime 우선, 없으면 tensorflow-lite로 폴백
try:
    import tflite_runtime.interpreter as tflite  # type: ignore
except ModuleNotFoundError:  # PyPI tensorflow>=2.5 설치 환경
    import tensorflow as tf  # type: ignore

    tflite = tf.lite  # pyright: ignore

# --------------------------------------------------------------------------------------
# 설정 & 로거
# --------------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("inference")

NUM_THREADS = int(os.getenv("NUM_THREADS", "4"))

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
    vec = np.asarray([float(sensor_data.get(k, 0.0)) for k in keys], dtype=np.float32)

    # 학습·양자화와 동일한 0~255 스케일 정규화
    mins = np.asarray([-100, 0, 950, -90, -180, 0], dtype=np.float32)
    maxs = np.asarray([100, 100, 1050, 90, 180, 1000], dtype=np.float32)
    vec = np.clip(vec, mins, maxs)
    vec = (vec - mins) * 255.0 / (maxs - mins)

    return np.asarray([vec], dtype=np.float32)

# --------------------------------------------------------------------------------------
# Helper to scale manual sensor value arrays (from --sensor_values)
# --------------------------------------------------------------------------------------

def _scale_sensor_vec(vec: np.ndarray) -> np.ndarray:
    mins = np.asarray([-100, 0, 950, -90, -180, 0], dtype=np.float32)
    maxs = np.asarray([100, 100, 1050, 90, 180, 1000], dtype=np.float32)
    vec = np.clip(vec, mins, maxs)
    vec = (vec - mins) * 255.0 / (maxs - mins)
    return vec.astype(np.float32)

# 수동 입력(쉼표 문자열) 파싱 → 학습 json이 없는 경우 fallback 용도
def _parse_sensor_values_manual(s: str | None) -> np.ndarray | None:
    if s is None:
        return None
    vals = [float(v.strip()) for v in s.split(",")]
    return np.asarray([vals], dtype=np.float32)

# --------------------------------------------------------------------------------------
# 보조 함수
# --------------------------------------------------------------------------------------

def _load_delegate(delegate_name: str | None):
    """EdgeTPU 등 하드웨어 가속용 delegate 로드."""

    if delegate_name is None:
        return None

    delegate_name = delegate_name.lower()
    if delegate_name == "edgetpu":
        try:
            return tflite.load_delegate("libedgetpu.so.1")
        except ValueError as e:  # 라이브러리 누락
            logger.warning("EdgeTPU delegate 로드 실패: %s", e)
            return None
    logger.warning("알 수 없는 delegate: %s", delegate_name)
    return None


def _new_interpreter(model_path: Path, delegate: Any | None = None):
    """주어진 모델 경로로 *tf.lite.Interpreter* 인스턴스를 생성한다."""

    kwargs: Dict[str, Any] = {"model_path": str(model_path), "num_threads": NUM_THREADS}
    if delegate is not None:
        kwargs["experimental_delegates"] = [delegate]
    return tflite.Interpreter(**kwargs)


def _prepare_input(pil: Image.Image, target_hw: Tuple[int, int], input_info: Dict[str, Any]) -> np.ndarray:
    """PIL 이미지를 모델 입력 요건(dtype, 정규화)에 맞게 ndarray로 변환.

    Parameters
    ----------
    pil : PIL.Image
        원본 이미지 (이미 RGB 변환 완료 상태).
    target_hw : (width, height)
        리사이즈 목표 크기.
    input_info : dict
        `Interpreter.get_input_details()[0]` 결과.
    Returns
    -------
    ndarray
        모델 입력 텐서 형태 (1, H, W, C)
    """

    w, h = target_hw
    img = pil.resize((w, h), Image.BILINEAR)
    arr = np.asarray(img)

    # 입력 DType 및 양자화 정보에 따라 전처리
    dtype = input_info["dtype"]
    if dtype == np.float32:
        arr = arr.astype(np.float32) / 255.0
    elif dtype == np.uint8:
        # uint8 양자화 모델 – 스케일/제로포인트 고려(선택적)
        scale, zero_point = input_info.get("quantization", (1.0, 0))
        if scale == 0:  # 일부 환경에서 (0, 0) 으로 세팅됨
            scale = 1.0 / 255.0
        arr = arr.astype(np.uint8)
        # 필요 시 float32 → uint8 변환 로직을 추가하려면 아래 주석 해제
        # arr = (arr.astype(np.float32) / 255.0 / scale + zero_point).astype(np.uint8)
    else:
        raise ValueError(f"지원되지 않는 입력 dtype: {dtype}")

    return np.expand_dims(arr, axis=0)


# --------------------------------------------------------------------------------------
# Visualization helpers
# --------------------------------------------------------------------------------------


def _overlay_mask_on_image(
    pil_img: Image.Image,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> Image.Image:
    """세그멘테이션 마스크를 원본 이미지 위에 반투명 색상으로 overlay.

    Parameters
    ----------
    pil_img : PIL.Image
        원본 RGB 이미지.
    mask : np.ndarray
        (H, W) 또는 (H, W, 1) 형태의 마스크. 값이 0이 아닌 픽셀을 overlay 대상으로 간주.
    color : (R, G, B)
        overlay 색상.
    alpha : float
        overlay 투명도 (0~1). 0은 완전 투명, 1은 완전 불투명.
    """

    # 마스크를 원본 크기로 리사이즈 (최근접)
    mask_img = Image.fromarray(mask.astype(np.uint8))
    mask_resized = mask_img.resize(pil_img.size, Image.NEAREST)

    pil_arr = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
    mask_arr = np.asarray(mask_resized, dtype=np.uint8)

    # boolean mask
    m = mask_arr > 0
    if m.ndim == 3:
        # 다중 클래스 argmax 결과 등 (H, W, 1) → (H, W)
        m = m[..., 0]

    overlay_arr = pil_arr.copy()
    overlay_color = np.array(color, dtype=np.uint8)
    overlay_arr[m] = (
        pil_arr[m].astype(np.float32) * (1.0 - alpha) + overlay_color.astype(np.float32) * alpha
    ).astype(np.uint8)

    return Image.fromarray(overlay_arr)


def process_image(
    image_data: bytes,
    seg_interp: Any,  # Interpreter 인스턴스
    cls_interp: Any,  # Interpreter 인스턴스 (또는 None)
    sensor_arr: np.ndarray | None = None,
) -> Dict[str, Any]:
    """단일 이미지 바이트스트림을 받아 세그멘테이션 & 분류 결과 반환.

    모델의 첫 번째 입력이 이미지인지 센서인지가 빌드 시점 옵션에 따라 달라질 수 있으므로,
    입력 텐서들의 shape 정보를 이용해 **이미지 입력(4-D)** 과 **센서 입력(≤2-D)**을
    자동으로 식별한다.
    """

    ts0 = time.time()

    with Image.open(io.BytesIO(image_data)) as pil:
        pil = pil.convert("RGB")

        # ------------------------------
        # Segmentation
        # ------------------------------
        seg_inputs = seg_interp.get_input_details()

        # 이미지 입력: 보통 shape (1, H, W, C) → len(shape) == 4
        img_in = next((d for d in seg_inputs if len(d["shape"]) >= 3), seg_inputs[0])

        # 센서 입력: 나머지 하나 (shape 길이가 2이거나 마지막 dim이 6)
        sensor_in = None
        if len(seg_inputs) > 1:
            sensor_in = next((d for d in seg_inputs if d is not img_in), None)

        # 이미지 전처리 및 입력
        seg_size = (img_in["shape"][2], img_in["shape"][1])  # (W, H)
        seg_arr = _prepare_input(pil, seg_size, img_in)

        seg_interp.set_tensor(img_in["index"], seg_arr)

        # 센서 입력 설정 (있을 경우)
        if sensor_in is not None:
            if sensor_arr is None:
                sensor_shape = sensor_in["shape"]
                sensor_arr_use = np.zeros(sensor_shape, dtype=np.float32)
            else:
                sensor_arr_use = sensor_arr.astype(np.float32)
                if sensor_arr_use.shape != tuple(sensor_in["shape"]):
                    sensor_arr_use = sensor_arr_use.reshape(sensor_in["shape"])

            # 센서 텐서를 인터프리터 기대 dtype으로 캐스팅 (uint8)
            sensor_arr_use = sensor_arr_use.astype(sensor_in["dtype"])

            seg_interp.set_tensor(sensor_in["index"], sensor_arr_use)

        seg_interp.invoke()

        seg_out = seg_interp.get_output_details()[0]
        mask = seg_interp.get_tensor(seg_out["index"])[0]
        # 후처리: 다중 클래스면 argmax, 단일 채널이면 threshold
        if mask.shape[-1] > 1:
            mask = np.argmax(mask, axis=-1).astype(np.uint8)
        else:
            if mask.dtype != np.float32:
                mask = mask.astype(np.float32)
            mask = (mask > 0.0).astype(np.uint8)

        # ------------------------------
        # Classification (선택)
        # ------------------------------
        cls_pred = None
        if cls_interp is not None:
            cls_in = cls_interp.get_input_details()[0]
            cls_size = (cls_in["shape"][2], cls_in["shape"][1])
            cls_arr = _prepare_input(pil, cls_size, cls_in)
            cls_interp.set_tensor(cls_in["index"], cls_arr)
            cls_interp.invoke()
            cls_out = cls_interp.get_output_details()[0]
            cls_pred = cls_interp.get_tensor(cls_out["index"])[0]

    return {
        "mask": mask,
        "cls_pred": cls_pred,
        "elapsed": time.time() - ts0,
    }

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Coral-DeepLab TFLite 추론 스크립트")
    p.add_argument("--input", required=True, help="입력 이미지 파일 또는 디렉터리")
    p.add_argument("--sensor_json", default=None, help="센서 정보 JSON 파일(선택)")
    p.add_argument("--seg_model", required=True, help="세그멘테이션 .tflite 경로")
    p.add_argument("--cls_model", default=None, help="분류 .tflite 경로(선택)")
    p.add_argument("--delegate", choices=["edgetpu"], default=None, help="사용 delegate")
    p.add_argument("--output_dir", default="inference_output", help="결과 저장 폴더")
    p.add_argument("--save_mask", action="store_true", help="세그멘테이션 마스크 PNG 저장 여부")
    p.add_argument("--ann_file", type=str, default=None, help="학습/검증에 사용된 COCO annotation JSON 경로")
    p.add_argument("--sensor_values", type=str, default=None, help="(옵션) 수동 센서 입력값 – JSON이 없을 때")
    return p


def main():
    args = _build_argparser().parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    delegate = _load_delegate(args.delegate)

    # ------------------------------------------------------------------
    # Annotation JSON에서 이미지별 sensor_info 매핑 생성
    # ------------------------------------------------------------------
    sensor_map: dict[str, np.ndarray] = {}
    if args.ann_file:
        logger.info("Loading annotations → %s", args.ann_file)
        with open(args.ann_file, "r") as f:
            ann_raw = json.load(f)
        img_list = ann_raw.get("images", [])
        for img in img_list:
            file_name = img.get("file_name")
            sensor_info = img.get("sensor_info", {})
            sensor_map[file_name] = _sensor_to_vec(sensor_info)

    # ------------------------------------------------------------------
    # 수동 / JSON 센서 입력 로딩
    # ------------------------------------------------------------------

    manual_sensor_arr: np.ndarray | None = None

    if getattr(args, "sensor_json", None):
        logger.info("Loading sensor JSON → %s", args.sensor_json)
        with open(args.sensor_json, "r") as f:
            sensor_raw = json.load(f)
        manual_sensor_arr = _sensor_to_vec(sensor_raw)
    else:
        raw_manual = _parse_sensor_values_manual(getattr(args, "sensor_values", None))
        if raw_manual is not None:
            manual_sensor_arr = _scale_sensor_vec(raw_manual)

    seg_interp = _new_interpreter(Path(args.seg_model), delegate)
    seg_interp.allocate_tensors()
    # 이미지 입력 텐서를 찾아서 shape 로깅
    seg_inputs = seg_interp.get_input_details()
    img_in = next((d for d in seg_inputs if len(d["shape"]) >= 3), seg_inputs[0])
    seg_shape = img_in["shape"]
    logger.info("Segmentation model loaded – input %dx%d", seg_shape[2], seg_shape[1])

    cls_interp = None
    if args.cls_model:
        cls_interp = _new_interpreter(Path(args.cls_model), delegate)
        cls_interp.allocate_tensors()
        cls_shape = cls_interp.get_input_details()[0]["shape"]
        logger.info("Classification model loaded – input %dx%d", cls_shape[2], cls_shape[1])

    # 입력 경로 확보
    in_paths: list[Path] = []
    p = Path(args.input)
    if p.is_dir():
        in_paths = sorted([f for f in p.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    else:
        in_paths = [p]

    for img_path in in_paths:
        with open(img_path, "rb") as f:
            img_sensor_arr = sensor_map.get(img_path.name, manual_sensor_arr)
            result = process_image(f.read(), seg_interp, cls_interp, img_sensor_arr)

        logger.info("Processed %s – %.3f s", img_path.name, result["elapsed"])

        # 마스크 저장 옵션
        if args.save_mask:
            mask = result["mask"]

            # 마스크 PNG 저장
            mask_img = Image.fromarray(mask)
            mask_img.save(out_dir / f"{img_path.stem}_mask.png")

            # Overlay 이미지 저장
            with Image.open(img_path) as orig_pil:
                overlay_img = _overlay_mask_on_image(orig_pil, mask)
                overlay_img.save(out_dir / f"{img_path.stem}_overlay.png")

        # 분류 결과 로그
        if result["cls_pred"] is not None:
            cls_pred = result["cls_pred"]
            logger.info("  → classification logits: %s", np.array2string(cls_pred, precision=3, separator=", "))


if __name__ == "__main__":
    main() 