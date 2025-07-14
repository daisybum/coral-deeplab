# Coral-DeepLab (센서 융합)

<img src="https://img.shields.io/badge/TensorFlow-2.x-orange"/>
<img src="https://img.shields.io/badge/Edge%20TPU-ready-brightgreen"/>

센서 정보(IMU·기상·GPS 등)와 RGB 이미지를 동시에 이용하는 모바일 세그멘테이션 네트워크 **Coral-DeepLab**의 TensorFlow/Keras 구현입니다.  
Google Coral Edge-TPU 및 Raspberry Pi(XNNPACK)에서 실시간 추론이 가능하도록 모델 구조를 경량화하고, 완전 정수(INT8) 양자화를 지원합니다.

---
## 목차
1. [주요 특징](#주요-특징)  
2. [디렉터리 구조](#디렉터리-구조)  
3. [설치](#설치)  
4. [데이터 준비](#데이터-준비)  
5. [학습](#학습)  
6. [전처리 파이프라인](#전처리-파이프라인)  
7. [체크포인트 이어-학습](#체크포인트-이어-학습)  
8. [모델 양자화 & 컴파일](#모델-양자화--컴파일)  
9. [추론](#추론)  
10. [참고 문헌](#참고-문헌)  

---
## 주요 특징
* **DeepLab V3 / V3+** 백본: MobileNet V2 (α 가중치 조절 지원)
* **SensorVisionFusion**: 6-차원 센서 벡터를 1×1×C 임베딩 후 feature map에 가중 합산
* **CBAM**(선택): Channel & Spatial Attention 모듈
* **완전 정수 양자화** → Edge-TPU / Lite RT(XNNPACK) 호환
* **DataLoader**: COCO-style JSON + `sensor_info` 필드를 지원하는 `tf.data` 파이프라인

---
## 디렉터리 구조
```
├── coral_deeplab/          # 라이브러리 소스
│   ├── _blocks.py          # ASPP · Decoder · Fusion 래퍼
│   ├── fusion.py           # SensorVisionFusion 레이어
│   ├── applications.py     # CoralDeepLabV3 / V3Plus 빌더
│   └── utils/              # 데이터셋 · 변환 헬퍼
├── config/                 # dataclass 기반 설정 모듈
├── data/                   # COCO JSON & 이미지 (사용자 준비)
├── example/                # 샘플 이미지/센서 JSON
├── checkpoints_tf/         # Keras 가중치 (*.keras)
├── compiler/               # Edge-TPU compiler wrapper
├── converter_*.py          # TFLite 변환 스크립트
├── inference*.py           # 추론 스크립트 (Keras / TFLite)
└── main.py                 # 학습 진입점
```

---
## 설치
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt            # TensorFlow 2.x, numpy ≤1.24 등
# OpenCV(헤드리스) 추가
pip install opencv-python-headless
```
Edge-TPU 컴파일을 위해서는 Coral SDK(`edgetpu_compiler v14+`)가 필요합니다.  
`compiler/compiler.sh` 스크립트로 자동 설치가 가능합니다.

---
## 데이터 준비
1. **이미지**: `data/images/` 폴더에 RGB 파일 배치  
2. **주석(JSON)**: COCO 포맷
```jsonc
{
  "images": [
    { "id": 1, "file_name": "xxx.jpg", "sensor_info": { /* 아래 예시 */ } },
    ...
  ],
  "annotations": [ { "image_id": 1, "category_id": 3, "segmentation": [...] }, ... ],
  "categories": [ {"id":1,"name":"car"}, ... ]
}
```
3. **sensor_info 필드** (키·순서 고정)
```jsonc
{
  "objectTemp": 24.8,    // °C  (-100~100)
  "humi": 79.3,          // %   (0~100)
  "pressure": 1013.1,    // hPa (950~1050)
  "latitude": 37.42,     // °   (-90~90)
  "longitude": 126.89,   // °   (-180~180)
  "height": 4.7          // m   (0~1000)
}
```
배경 라벨은 **ID 0** 으로 암묵적으로 할당됩니다.

---
## 학습
```bash
python main.py \
  --train_annotations data/COCO/train.json \
  --val_annotations   data/COCO/valid.json \
  --train_images      data/images \
  --val_images        data/images \
  --model deeplabv3plus --batch_size 64 --epochs 200
```
옵션
* `--model` : `deeplabv3` | `deeplabv3plus`
* `--lr`    : 초기 학습률 (기본 1e-4)
* `--resume`: `checkpoints_tf/epoch_XXX.keras` 이어-학습

---
## 전처리 파이프라인
| 항목 | dtype | 범위 | 비고 |
|------|-------|------|------|
| 이미지 | float32 | 0 – 1 | `OpenCV BGR → RGB /255.0` |
| 센서   | float32 | 0 – 255 | `mins/maxs` 테이블로 클립 & 스케일 |

모델 입력: `(image, sensors)`  👈 두 텐서를 리스트로 전달합니다.

### 센서 전처리 함수 (매우 중요) `_sensor_to_vec`

데이터 로더와 추론 스크립트에 공통으로 쓰이는 함수로, 센서 JSON(또는 dict) → `float32 (6,)` 벡터를 생성합니다.

1. **필드 추출**  
   `objectTemp`, `humi`, `pressure`, `latitude`, `longitude`, `height` 여섯 키를 고정 순서로 읽어옵니다.  
   누락된 값은 0.0 으로 대체합니다.

2. **범위 클리핑 & 0-255 정규화**  
   | 항목 | min | max |
   |------|-----|-----|
   | objectTemp | –100 | 100 |
   | humi       |   0  | 100 |
   | pressure   | 950  | 1050|
   | latitude   | –90  | 90  |
   | longitude  | –180 | 180 |
   | height     | 0    | 1000|

   값이 위 범위를 넘으면 `np.clip`으로 잘라낸 뒤  
   `(value-min) × 255 / (max-min)` 식을 사용해 **0~255 float32** 값으로 변환합니다.

3. **배치 차원 추가**  
   학습 시에는 `(B, 6)`, 추론 스크립트에서는 `(1, 6)` 형태로 래핑해 모델에 전달합니다.

이 과정을 통해 학습·양자화·추론 모든 단계에서 센서 입력 분포가 완전히 일치하게 됩니다.

---
## 체크포인트 이어-학습
```bash
python main.py --batch_size 64 --epochs 500 \
               --resume checkpoints_tf/epoch_144.keras
```
`epoch_144.keras` 까지 학습된 모델을 불러와 **epoch 145** 부터 이어서 학습합니다.

---
## 모델 양자화 & 컴파일
### Raspberry Pi (XNNPACK)
```bash
python converter_rpi_sensor.py            # seg_model_int8.tflite 생성
```
### Edge-TPU
```bash
bash compiler/compiler.sh seg_model_int8.tflite
# → seg_model_int8_edgetpu.tflite
```

### 대표 데이터셋 스케일 주의
> **대표 데이터셋 스케일 주의**  
>   • **이미지**: `rand()  →  0 ~ 1` *(변환기에서 scale≈1/255 추정)*  
>   • **센서**  : `rand()*255 → 0 ~ 255` *(scale≈1 추정)*  
>   이렇게 구성해야 완전 정수 양자화 모델이 이미지/센서 각각에 맞는 scale 값을 저장합니다. 0 ~ 255 `uint8` 이미지를 그대로 넣어도 `(val−0)×1/255` 로 0 ~ 1 로 복원되어 Keras 결과와 거의 동일해집니다.

### CPU 기본 실행 및 TPU 선택적 사용
* 변환·양자화 과정은 **CPU** 만으로 수행됩니다. CUDA / GPU가 없어도 문제 없습니다.
* 추론 역시 기본적으로 **CPU(XNNPACK delegate)** 경로를 사용하며, `--delegate edgetpu` 플래그를 줘야만 Edge-TPU 가속이 활성화됩니다.

---
## 추론
### Keras 체크포인트
```bash
python inference_with_keras.py \
  --input data/images/example.jpg \
  --sensor_json data/images/example.json \
  --ckpt checkpoints_tf/epoch_187.keras \
  --save_mask --output_dir results
```
### TFLite 모델 (CPU/XNNPACK 또는 Edge-TPU)
```bash
python inference.py \
  --input data/images/example.jpg \
  --seg_model seg_model_int8_edgetpu.tflite \
  --delegate edgetpu --save_mask
```

---
## 테스트 세트 성능

다음 표는 *test* split 전체에 대해 `test_with_keras.py` 및 `test_with_int8.py` 로 측정한 주요 지표입니다.

| 모델 | Pixel Accuracy | Mean IoU | Mean Dice | Frequency-Weighted IoU |
|------|---------------|----------|-----------|------------------------|
| **Keras (float32)** | **0.9302** | **0.7922** | **0.8809** | **0.8760** |
| **TFLite INT8** | 0.9271 | 0.7843 | 0.8757 | 0.8712 |

> *참고*: Keras 모델 대비 INT8 TFLite 모델은 양자화 손실로 인해 mIoU 약 **0.8pt** 정도, Pixel Accuracy 약 **0.3pt** 감소했지만, 전반적으로 유사한 성능을 유지합니다. Edge-TPU delegate 사용 시 최대 **x10** 배의 추론 속도 향상을 기대할 수 있습니다.

---
## 참고 문헌
* Chen et al., *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*, 2018  
* Woo et al., *CBAM: Convolutional Block Attention Module*, 2018  
* Google Coral, [Semantic Segmentation with Edge-TPU](https://coral.ai/models/)