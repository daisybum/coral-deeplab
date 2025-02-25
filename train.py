import os
import json
import math
import numpy as np
import tensorflow as tf

# 1. 데이터 로딩: COCO JSON에서 이미지 경로 및 마스크 생성
# ===========================================================
import pycocotools.coco as coco_tools  # pycocotools 설치 필요: pip install pycocotools

def load_coco_dataset(annotation_path, images_dir):
    """
    COCO 형식의 어노테이션 JSON을 로드하여 이미지 파일 경로와 
    해당 세그멘테이션 마스크를 생성합니다.
    """
    coco = coco_tools.COCO(annotation_path)
    image_ids = coco.getImgIds()  # 모든 이미지 ID 목록
    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_path = os.path.join(images_dir, img_info['file_name'])
        # 이미지 너비와 높이
        width, height = img_info['width'], img_info['height']
        # 해당 이미지의 모든 어노테이션 불러오기
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        # 초기 배경을 0으로 하는 마스크 배열 생성
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in anns:
            # segmentation을 binary mask로 변환
            mask_ann = coco.annToMask(ann)  # 0 또는 1 값을 가지는 mask
            # mask 영역에 해당 객체의 카테고리 ID 할당
            mask[mask_ann == 1] = ann['category_id']
        yield file_path, mask

# argparse를 사용하여 입력 경로와 파라미터를 받을 수 있게 구성
import argparse
parser = argparse.ArgumentParser(description="Train CoralDeepLabV3 on COCO-format dataset")
parser.add_argument("--train_annotations", type=str, required=True, help="학습용 COCO JSON 어노테이션 경로")
parser.add_argument("--train_images", type=str, required=True, help="학습 이미지 디렉토리 경로")
parser.add_argument("--val_annotations", type=str, default=None, help="검증용 COCO JSON 어노테이션 경로 (선택)")
parser.add_argument("--val_images", type=str, default=None, help="검증 이미지 디렉토리 경로 (선택)")
parser.add_argument("--batch_size", type=int, default=8, help="배치 크기")
parser.add_argument("--epochs", type=int, default=20, help="에포크 수")
parser.add_argument("--output_dir", type=str, default="checkpoints", help="체크포인트 저장 디렉토리")
args = parser.parse_args()

train_ann_path = args.train_annotations
train_img_dir = args.train_images
val_ann_path = args.val_annotations
val_img_dir = args.val_images
batch_size = args.batch_size
epochs = args.epochs
output_dir = args.output_dir

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# 학습 데이터 tf.data.Dataset 생성 (generator 활용)
train_dataset = tf.data.Dataset.from_generator(
    lambda: load_coco_dataset(train_ann_path, train_img_dir),
    output_types=(tf.string, tf.uint8),
    output_shapes=((), (None, None))
)

# 검증 데이터 tf.data.Dataset 생성 (있을 경우)
val_dataset = None
if val_ann_path and val_img_dir:
    val_dataset = tf.data.Dataset.from_generator(
        lambda: load_coco_dataset(val_ann_path, val_img_dir),
        output_types=(tf.string, tf.uint8),
        output_shapes=((), (None, None))
    )

# 2. 데이터 전처리 및 증강: 이미지 디코딩, 크기변환, 정규화 + Random Flip, Crop, Color Jitter
# ======================================================================
def parse_image_mask(image_path, mask):
    """이미지 파일을 디코딩하고 정규화하며, 마스크를 Tensor로 변환"""
    # 이미지 파일 로드 및 디코딩 (채널=3)
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32) / 255.0  # 0~1 정규화
    # (MobilenetV2 사전학습 사용 시 이미지 정규화를 [-1,1]로 조정 필요)
    mask = tf.cast(mask, tf.int32)
    return image, mask

def augment_image_mask(image, mask):
    """훈련용 데이터 증강: 좌우반전, 랜덤크롭, 밝기/대비 변화 적용"""
    # 마스크에 채널 차원 추가 (이미지와 함께 연산하기 위함)
    mask = mask[..., tf.newaxis]  # (H, W, 1)
    # 2.1 무작위 좌우 반전
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    # 2.2 랜덤 크롭: 이미지를 513x513으로 랜덤 자르기 (필요시 패딩)
    original_height = tf.shape(image)[0]
    original_width = tf.shape(image)[1]
    # 높이/너비가 513보다 작으면 패딩하여 크기 확보
    pad_height = tf.maximum(513 - original_height, 0)
    pad_width = tf.maximum(513 - original_width, 0)
    if pad_height > 0 or pad_width > 0:
        image = tf.pad(image, [[0, pad_height], [0, pad_width], [0, 0]], constant_values=0.0)
        mask = tf.pad(mask, [[0, pad_height], [0, pad_width], [0, 0]], constant_values=0)
    # 패딩 후 실제 크기
    new_height = tf.shape(image)[0]
    new_width = tf.shape(image)[1]
    # 자를 위치의 시작 좌표를 무작위로 선택
    offset_height = tf.random.uniform((), 0, new_height - 513 + 1, dtype=tf.int32)
    offset_width = tf.random.uniform((), 0, new_width - 513 + 1, dtype=tf.int32)
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, 513, 513)
    mask = tf.image.crop_to_bounding_box(mask, offset_height, offset_width, 513, 513)
    # 2.3 색상 변화: 밝기 및 대비 무작위 조절
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.2)   # 밝기 조절
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # 대비 조절
    # 마스크 채널 차원 제거
    mask = tf.squeeze(mask, axis=-1)
    return image, mask

def resize_image_mask(image, mask):
    """검증용 데이터 전처리: 이미지와 마스크를 513x513으로 리사이즈"""
    image = tf.image.resize(image, (513, 513), method=tf.image.ResizeMethod.BILINEAR)
    # 마스크는 최근접 이웃 방식으로 리사이즈
    mask = tf.image.resize(mask[..., tf.newaxis], (513, 513), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.squeeze(mask, axis=-1)
    return image, tf.cast(mask, tf.int32)

# 데이터 파이프라인 구성
AUTOTUNE = tf.data.AUTOTUNE
# 학습 데이터셋: 파싱 -> 증강 -> 셔플 -> 배치 -> 프리페치
train_dataset = (train_dataset
                 .map(parse_image_mask, num_parallel_calls=AUTOTUNE)
                 .map(augment_image_mask, num_parallel_calls=AUTOTUNE)
                 .shuffle(buffer_size=1000)
                 .batch(batch_size)
                 .prefetch(AUTOTUNE))
# 검증 데이터셋: 파싱 -> 리사이즈 -> 배치 -> 프리페치 (증강 없음)
if val_dataset is not None:
    val_dataset = (val_dataset
                   .map(parse_image_mask, num_parallel_calls=AUTOTUNE)
                   .map(resize_image_mask, num_parallel_calls=AUTOTUNE)
                   .batch(batch_size)
                   .prefetch(AUTOTUNE))

# 클래스 개수 정의 (배경 포함)
# COCO JSON의 categories 필드 사용: 최대 category_id + 1 을 클래스 수로 설정
with open(train_ann_path, 'r') as f:
    train_json = json.load(f)
category_ids = [cat["id"] for cat in train_json.get("categories", [])]
num_classes = (max(category_ids) + 1) if category_ids else 1

# 3. 모델 학습 구성: 모델 로드, 손실/옵티마이저/학습률 스케줄, 콜백 정의
# ================================================================
# CoralDeepLabV3 모델 로드 (MobileNet-v2 백본, 입력크기 513, 클래스 수 지정)
try:
    import coral_deeplab as cdl
    model = cdl.applications.CoralDeepLabV3(input_shape=(513, 513, 3),
                                            backbone='MobileNetV2',
                                            classes=num_classes)
except ImportError:
    # coral_deeplab 패키지가 없는 경우, 사용자 정의 CoralDeepLabV3 구현 필요
    raise ImportError("CoralDeepLabV3 모델을 찾을 수 없습니다. coral_deeplab 패키지 설치 또는 모델 구현을 확인하세요.")

# 손실 함수 설정: 다중 클래스 세그멘테이션에는 sparse_categorical_crossentropy 사용&#8203;:contentReference[oaicite:1]{index=1} 
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# (참고: 불균형 데이터셋의 경우 Dice Loss를 커스텀하여 사용할 수도 있습니다.)

# 최적화 알고리즘 및 학습률 스케줄: Adam + PolynomialDecay 스케줄
steps_per_epoch = math.ceil(len(train_json["images"]) / batch_size)
# Polynomial decay: 초기 lr에서 최종 lr로 선형 감소 (power=1.0, 에포크 전체에 걸쳐)
initial_lr = 1e-3
end_lr = 1e-5
decay_steps = steps_per_epoch * epochs
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=initial_lr,
                                                            decay_steps=decay_steps,
                                                            end_learning_rate=end_lr,
                                                            power=1.0)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# (참고: SGD(momentum=0.9)으로 변경할 수도 있음:
#  optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9))

# 평가 지표: MeanIoU (교차엔트로피 기준 픽셀 정확도 외에 IoU 사용)
metrics = [tf.keras.metrics.MeanIoU(num_classes=num_classes)]

# 모델 컴파일
model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

# 체크포인트 콜백: 지정한 output_dir에 에포크별 가중치 저장
ckpt_path = os.path.join(output_dir, "deeplabv3_epoch{epoch:02d}.h5")
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_weights_only=True, save_freq='epoch')
# TensorBoard 콜백: logs 디렉토리에 훈련 기록 저장
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)

# 4. 모델 학습 실행: 학습 시작 및 (검증 수행 시) 매 에포크마다 평가
# ==============================================================
print("Training started... (epochs: {}, batch_size: {})".format(epochs, batch_size))
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset if val_dataset is not None else None,
    callbacks=[checkpoint_cb, tensorboard_cb]
)
print("Training completed.")
