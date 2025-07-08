"""
convert_to_litert_int8_sensor.py
─────────────────────────────────
DeepLabV3+ (Image + Sensor) Keras 가중치를 → INT8 양자화 TFLite
대상: Raspberry Pi 5 (ARM, LiteRT + XNNPACK)

이 스크립트는 이미지 입력과 1-D 센서 벡터(예: 6 차원)를 동시에 받는
CoralDeepLabV3Plus 모델을 TFLite INT8 형식으로 변환합니다.
"""

from __future__ import annotations

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.models import Model
import coral_deeplab as cdl  # 프로젝트 루트에 위치한 커스텀 패키지

# --------------------------------------------------------------------------------------
# 설정 값
# --------------------------------------------------------------------------------------
NUM_CLASSES: int = 6      # 세그멘테이션 클래스 수
SENSOR_DIM: int = 6       # 센서 벡터 길이 (데이터셋과 동일하게 맞추세요)
FIXED_SIZE = (513, 513)   # 입력 이미지 해상도 `(H, W)

INPUT_PATH = "checkpoints_tf/epoch_110.keras"  # 학습된 Keras 가중치
OUTPUT_PATH = "seg_model_sensor_int8.tflite"  # 변환 후 저장 파일

# --------------------------------------------------------------------------------------
# 1) 가중치 로드 및 모델 구성
# --------------------------------------------------------------------------------------
print("[1/3] Loading base model and weights …")
base_model = cdl.applications.CoralDeepLabV3Plus(
    input_shape=(*FIXED_SIZE, 3),
    n_classes=NUM_CLASSES,
    sensor_dim=SENSOR_DIM,
)
# 학습 당시 모델이 센서 경로 없이 저장되었을 경우를 대비해
try:
    base_model.load_weights(INPUT_PATH)
except ValueError as e:
    print("[warn] layer mismatch – attempting partial load with skip_mismatch=True …")
    base_model.load_weights(INPUT_PATH, by_name=True, skip_mismatch=True)

# 입력 placeholder 정의 (고정 해상도 + 센서)
img_in = Input(shape=(*FIXED_SIZE, 3), name="image")
sensor_in = Input(shape=(SENSOR_DIM,), name="sensors")

# 모델 실행 → 크기 보정 및 소프트맥스 래핑
logits = base_model([img_in, sensor_in])  # (B, 513, 513, N)
# DeepLabV3+ 는 이미 513×513 출력이므로 resize 생략 가능하지만,
# 안전을 위해 동일 해상도로 맞춰 둡니다.
logits = tf.image.resize(logits, FIXED_SIZE, method="bilinear")
outputs = Activation("softmax", name="pred_softmax")(logits)

fixed_model = Model(inputs=[img_in, sensor_in], outputs=outputs, name="deeplabv3_sensor_fixed")
print(fixed_model.summary())

# --------------------------------------------------------------------------------------
# 2) TFLite 변환기 설정
# --------------------------------------------------------------------------------------
print("[2/3] Configuring TFLite converter …")
converter = tf.lite.TFLiteConverter.from_keras_model(fixed_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 대표 데이터셋: 무작위 샘플(또는 실제 샘플)로 스케일 보정

def representative_dataset():
    """Generator yielding dict that maps **input layer names** to sample tensors.

    TFLite quantizer is sensitive to input ordering; using a dict ensures each
    sample is fed to the correct placeholder ("image" vs "sensors").
    """
    for _ in range(250):
        yield {
            "image": np.random.rand(1, *FIXED_SIZE, 3).astype(np.float32),
            "sensors": np.random.rand(1, SENSOR_DIM).astype(np.float32),
        }

converter.representative_dataset = representative_dataset

# Pi 5 (Lite Runtime + XNNPACK) 정수 연산만 사용
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8   # 모든 입력 타입
converter.inference_output_type = tf.uint8

# --------------------------------------------------------------------------------------
# 3) 변환 & 저장
# --------------------------------------------------------------------------------------
print("[3/3] Converting … (this may take a few minutes)")
tflite_model = converter.convert()

os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)
print(f"TFLite(INT8) saved → {OUTPUT_PATH}")
