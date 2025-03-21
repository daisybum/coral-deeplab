import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import coral_deeplab as cdl

NUM_CLASSES = 6
FIXED_SIZE = (513, 513)

input_path = "deeplabv3_epoch500.h5"
output_path = "model_quant_fixed.tflite"

# 1) 모델 초기화 (input_shape 명시)
base_model = cdl.applications.CoralDeepLabV3Plus(input_shape=(513, 513, 3), n_classes=NUM_CLASSES)

# 2) 가중치 로드
base_model.load_weights(input_path)

# 3) 입력 tensor 정의 및 base_model 연결 (명시적 고정)
inputs = Input(shape=(513, 513, 3))
x = base_model(inputs)
output = tf.image.resize(x, FIXED_SIZE, method='bilinear')
output = tf.keras.layers.Activation('softmax')(output)

fixed_model = Model(inputs=inputs, outputs=output)

# base_model 가중치와 연결된 fixed_model 가중치는 이미 로드된 상태임

# 4) TFLite 변환기 설정 (정수 양자화)
converter = tf.lite.TFLiteConverter.from_keras_model(fixed_model)
converter.experimental_new_converter = False
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    for _ in range(100):
        data = np.random.rand(1, 513, 513, 3).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_data_gen

# Edge TPU 필수 설정
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# 모델 변환 후 저장
quantized_tflite_model = converter.convert()

with open(output_path, 'wb') as f:
    f.write(quantized_tflite_model)
