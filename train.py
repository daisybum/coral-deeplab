import os
import json
import math
import numpy as np
import tensorflow as tf
from PIL import Image
import pycocotools.coco as coco_tools  # pycocotools 설치 필요: pip install pycocotools

# 마스크 ID 리매핑 함수
def remap_mask(mask, id_mapping):
    remapped = np.zeros_like(mask, dtype=np.uint8)
    for orig_id, new_id in id_mapping.items():
        remapped[mask == orig_id] = new_id
    return remapped

# COCO 데이터셋 로드 함수
def load_coco_dataset(annotation_path, images_dir):
    coco = coco_tools.COCO(annotation_path)
    image_ids = coco.getImgIds()
    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_path = os.path.join(images_dir, img_info['file_name'])
        with Image.open(file_path) as img:
            width, height = img.size
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in anns:
            mask_ann = coco.annToMask(ann)
            mask[mask_ann == 1] = ann['category_id']
        mask = remap_mask(mask, id_mapping)
        yield file_path, mask

# 커스텀 MeanIoU 메트릭
class MeanIoUWithArgmax(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        return super().update_state(y_true, y_pred, sample_weight)

# 명령줄 인자 파싱
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

# 데이터셋 생성
train_dataset = tf.data.Dataset.from_generator(
    lambda: load_coco_dataset(train_ann_path, train_img_dir),
    output_types=(tf.string, tf.uint8),
    output_shapes=((), (None, None))
)

val_dataset = None
if val_ann_path and val_img_dir:
    val_dataset = tf.data.Dataset.from_generator(
        lambda: load_coco_dataset(val_ann_path, val_img_dir),
        output_types=(tf.string, tf.uint8),
        output_shapes=((), (None, None))
    )

# 데이터 전처리 및 증강 함수
def parse_image_mask(image_path, mask):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.int32)
    return image, mask

def augment_image_mask(image, mask):
    # 마스크를 3D 텐서로 변환 (높이, 너비, 1)
    mask = mask[..., tf.newaxis]
    
    # 1. 좌우 반전
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    
    # 2. 상하 반전
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    
    # 3. 무작위 회전 (0, 90, 180, 270도)
    k = tf.random.uniform((), 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    mask = tf.image.rot90(mask, k=k)
    
    # 4. 밝기 조정
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.2)
    
    # 5. 대비 조정
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # 6. 색상 조정
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_hue(image, max_delta=0.1)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    
    # 7. 이미지와 마스크를 513x513으로 리사이즈
    image = tf.image.resize(image, [513, 513], method='bilinear')
    mask = tf.image.resize(mask, [513, 513], method='nearest')
    
    # 마스크를 다시 2D로 변환
    mask = tf.squeeze(mask, axis=-1)
    
    return image, mask

def resize_image_mask(image, mask):
    mask = tf.cast(mask, tf.int32)
    image = tf.image.resize(image, (513, 513), method=tf.image.ResizeMethod.BILINEAR)
    mask = tf.keras.layers.Resizing(513, 513, interpolation='nearest')(mask[..., tf.newaxis])
    mask = tf.squeeze(mask, axis=-1)
    return image, tf.cast(mask, tf.int32)

# 데이터 파이프라인 구성
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = (train_dataset
                 .map(parse_image_mask, num_parallel_calls=AUTOTUNE)
                 .map(augment_image_mask, num_parallel_calls=AUTOTUNE)
                 .map(resize_image_mask, num_parallel_calls=AUTOTUNE)
                 .shuffle(buffer_size=1000)
                 .batch(batch_size)
                 .prefetch(AUTOTUNE))

if val_dataset is not None:
    val_dataset = (val_dataset
                   .map(parse_image_mask, num_parallel_calls=AUTOTUNE)
                   .map(resize_image_mask, num_parallel_calls=AUTOTUNE)
                   .batch(batch_size)
                   .prefetch(AUTOTUNE))

# 클래스 수 정의
with open(train_ann_path, 'r') as f:
    train_json = json.load(f)

unique_ids = sorted({cat["id"] for cat in train_json["categories"]})
if 0 not in unique_ids:
    unique_ids = [0] + unique_ids
id_mapping = {orig_id: new_id for new_id, orig_id in enumerate(unique_ids)}
num_classes = len(unique_ids)

# 모델 설정
try:
    import coral_deeplab as cdl
    model = cdl.applications.CoralDeepLabV3Plus(input_shape=(513, 513, 3),
                                            n_classes=num_classes)
except ImportError:
    raise ImportError("CoralDeepLabV3 모델을 찾을 수 없습니다. coral_deeplab 패키지 설치 또는 모델 구현을 확인하세요.")

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
steps_per_epoch = math.ceil(len(train_json["images"]) / batch_size)
initial_lr = 1e-4
end_lr = 1e-6
decay_steps = steps_per_epoch * epochs
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=initial_lr,
                                                            decay_steps=decay_steps,
                                                            end_learning_rate=end_lr,
                                                            power=1.0)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
metrics = [tf.keras.metrics.MeanIoU(num_classes=num_classes)]

model.compile(optimizer=optimizer, loss=loss_fn, metrics=[MeanIoUWithArgmax(num_classes=num_classes)])

# 콜백 정의
ckpt_path = os.path.join(output_dir, "deeplabv3_epoch{epoch:02d}.h5")
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_weights_only=True, save_freq='epoch')
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)

# 모델 학습
print("Training started... (epochs: {}, batch_size: {})".format(epochs, batch_size))
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset if val_dataset is not None else None,
    callbacks=[checkpoint_cb, tensorboard_cb]
)
print("Training completed.")