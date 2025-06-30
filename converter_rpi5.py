"""
convert_to_litert_int8.py
────────────────────────────────────────────
· DeepLabV3+ Keras 가중치를 → INT8 양자화 TFLite
· 추론 대상: Raspberry Pi 5 ARM (LiteRT + XNNPACK)
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import coral_deeplab as cdl   # 사용자 커스텀 모듈

NUM_CLASSES  = 6
FIXED_SIZE   = (513, 513)
INPUT_PATH   = "deeplabv3_epoch500.h5"
OUTPUT_PATH  = "seg_model_int8.tflite"

# ── 1) 모델 로드 ─────────────────────────────────────
base_model = cdl.applications.CoralDeepLabV3Plus(
    input_shape=(*FIXED_SIZE, 3), n_classes=NUM_CLASSES
)
base_model.load_weights(INPUT_PATH)

# 고정 크기 입력/출력 라핑
inputs  = Input(shape=(*FIXED_SIZE, 3))
x       = base_model(inputs)
x       = tf.image.resize(x, FIXED_SIZE, method="bilinear")
outputs = tf.keras.layers.Activation("softmax")(x)
fixed_model = Model(inputs, outputs)

# ── 2) TFLite 변환기 설정 ────────────────────────────
converter = tf.lite.TFLiteConverter.from_keras_model(fixed_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 대표 데이터셋 (양자화 스케일 산출용)
def rep_data():
    for _ in range(1000):
        yield [np.random.rand(1, *FIXED_SIZE, 3).astype(np.float32)]

converter.representative_dataset = rep_data

# Pi ARM + XNNPACK INT8 호환 설정
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.uint8   # 입력 dtype
converter.inference_output_type = tf.uint8   # 출력 dtype

# ── 3) 변환 & 저장 ───────────────────────────────────
tflite_model = converter.convert()
with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print(f"TFLite(INT8) saved → {OUTPUT_PATH}")
