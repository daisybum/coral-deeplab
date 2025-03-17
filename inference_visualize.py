import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pycocotools.coco import COCO

import coral_deeplab as cdl

# -----------------------------
# 사용자 설정 부분
# -----------------------------
ANNOTATION_FILE = "/workspace/merged_all/test_annotations.coco.json"
IMAGE_DIR = "/workspace/merged_all"
MODEL_WEIGHTS = "checkpoints/deeplabv3_epoch500.h5"
OUTPUT_DIR = "output_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 세그멘테이션 클래스 정의 (class_id: (class_name, [B, G, R]))
# 여기서는 예시로 5개의 상태(dry, humid, slush, snow, wet)를 지정
CLASS_INFO = {
    1: ("dry",   [113, 193, 255]),  # (B, G, R) 예시
    2: ("humid", [255, 219, 158]),
    3: ("slush", [125, 255, 238]),
    4: ("snow",  [255, 255, 255]),
    5: ("wet",   [255, 61, 61])
}
# 배경은 class_id=0으로 가정, 검정색으로 처리할 예정
BACKGROUND_COLOR = [0, 0, 0]

# -----------------------------
# 모델 로드 (사용자 모델에 맞춰 수정)
# -----------------------------
# 예시: 사용자 정의 함수 혹은 이미 저장된 모델 로드
def create_deeplabv3_model(num_classes=6, input_shape=(513, 513, 3)):
    model = cdl.applications.CoralDeepLabV3Plus(input_shape=(513, 513, 3),
                                            n_classes=num_classes)
    return model

try:
    # 사용자 구현 모델 불러오기
    model = create_deeplabv3_model(num_classes=len(CLASS_INFO)+1)  # +1은 배경 클래스
    model.load_weights(MODEL_WEIGHTS)
except NotImplementedError:
    print("[!] DeepLabV3 모델 생성이 구현되지 않아 임시로 예시 메시지를 표시합니다.")
    # 실제로는 위 함수를 구현 후, weights를 불러와야 합니다.
    model = None

# -----------------------------
# COCO 어노테이션 로드
# -----------------------------
coco = COCO(ANNOTATION_FILE)
image_ids = coco.getImgIds()

# -----------------------------
# 시각화 함수
# -----------------------------
def visualize_result(orig_img_bgr, pred_mask, file_name):
    """
    orig_img_bgr: 원본(BGR) 이미지 (numpy array, shape=(H, W, 3))
    pred_mask   : 모델 예측 마스크 (int형, shape=(H, W)), 각 픽셀은 class_id
    file_name   : 결과 저장 파일명
    """
    # 원본 이미지를 RGB로 변환 (Matplotlib는 RGB 사용)
    orig_img_rgb = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)

    # 컬러 마스크 만들기
    # 배경(0)인 픽셀은 검정색, 그 외는 CLASS_INFO의 색상
    color_mask = np.zeros_like(orig_img_bgr, dtype=np.uint8)
    # 배경 채우기
    color_mask[pred_mask == 0] = BACKGROUND_COLOR
    # 각 클래스별로 채우기
    for cls_id, info in CLASS_INFO.items():
        class_name, bgr_color = info
        color_mask[pred_mask == cls_id] = bgr_color
    
    # Overlay 이미지 생성 (50% blending)
    overlay_img = orig_img_bgr.copy()
    fg_mask = (pred_mask != 0)  # 배경이 아닌 곳만 합성
    overlay_img[fg_mask] = cv2.addWeighted(
        overlay_img[fg_mask],
        0.5,
        color_mask[fg_mask],
        0.5,
        0
    )
    overlay_img_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

    # Matplotlib으로 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1) 원본
    axes[0].imshow(orig_img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 2) 세그멘테이션 마스크 (BGR -> RGB 변환)
    mask_rgb = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)
    axes[1].imshow(mask_rgb)
    axes[1].set_title("Segmentation Mask")
    axes[1].axis("off")

    # 3) 오버레이
    axes[2].imshow(overlay_img_rgb)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    # 범례(legend) 만들기
    patches = []
    for cls_id, (class_name, bgr_color) in CLASS_INFO.items():
        rgb_color = tuple(v/255.0 for v in bgr_color)  # 0~1 범위
        patch = mpatches.Patch(color=rgb_color, label=class_name)
        patches.append(patch)

    # 범례를 figure 오른쪽 빈 공간에 배치
    # loc='center left', bbox_to_anchor=(1.0, 0.5) 등 다양하게 조절 가능
    fig.legend(
        handles=patches,
        loc='center right',
        title="Classes"
    )

    plt.tight_layout()
    
    # 결과 저장
    save_path = os.path.join(OUTPUT_DIR, file_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

# -----------------------------
# 추론(Inference) & 시각화
# -----------------------------
if model is not None:
    for img_id in image_ids[:5]:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        img_path = os.path.join(IMAGE_DIR, file_name)

        # 원본 이미지 읽기 (BGR)
        orig_img_bgr = cv2.imread(img_path)
        if orig_img_bgr is None:
            continue

        # (옵션) 모델 입력 사이즈에 맞게 리사이즈
        h, w = orig_img_bgr.shape[:2]
        input_img = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (513, 513))  # 예시
        input_img = input_img.astype(np.float32) / 255.0
        input_tensor = tf.expand_dims(input_img, axis=0)

        # 모델 추론
        pred_logits = model.predict(input_tensor)            # (1, 513, 513, num_classes)
        pred_mask_513 = tf.argmax(pred_logits, axis=-1)[0]   # (513, 513)
        pred_mask_513 = pred_mask_513.numpy().astype(np.uint8)

        # 원본 해상도 복원
        pred_mask = cv2.resize(pred_mask_513, (w, h), interpolation=cv2.INTER_NEAREST)

        # 결과 시각화
        result_name = os.path.splitext(file_name)[0] + "_result.png"
        visualize_result(orig_img_bgr, pred_mask, result_name)

else:
    print("딥랩 모델이 정의되지 않았습니다. create_deeplabv3_model 함수를 구현 후, 가중치를 로드하세요.")
