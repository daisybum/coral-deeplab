import os
import json
import math
import numpy as np
import tensorflow as tf
from PIL import Image
import pycocotools.coco as coco_tools
import argparse
from sklearn.metrics import average_precision_score

# 마스크 ID 리매핑 함수
def remap_mask(mask, id_mapping):
    remapped = np.zeros_like(mask, dtype=np.uint8)
    for orig_id, new_id in id_mapping.items():
        remapped[mask == orig_id] = new_id
    return remapped

# COCO 데이터셋 로드 함수 (id_mapping 인자 추가)
def load_coco_dataset(annotation_path, images_dir, id_mapping):
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

# 이미지와 마스크 파싱 (테스트용)
def parse_image_mask(image_path, mask):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.int32)
    return image, mask

# 이미지와 마스크 리사이즈 (테스트용)
def resize_image_mask(image, mask):
    mask = tf.cast(mask, tf.int32)
    image = tf.image.resize(image, (513, 513), method=tf.image.ResizeMethod.BILINEAR)
    mask = tf.keras.layers.Resizing(513, 513, interpolation='nearest')(mask[..., tf.newaxis])
    mask = tf.squeeze(mask, axis=-1)
    return image, tf.cast(mask, tf.int32)

def main(args):
    # 테스트 데이터셋 어노테이션 로드
    with open(args.test_annotations, 'r') as f:
        test_json = json.load(f)
    
    # COCO 카테고리 ID 매핑 (학습 시와 동일하게)
    unique_ids = sorted({cat["id"] for cat in test_json["categories"]})
    if 0 not in unique_ids:
        unique_ids = [0] + unique_ids
    id_mapping = {orig_id: new_id for new_id, orig_id in enumerate(unique_ids)}
    num_classes = len(unique_ids)
    print("총 클래스 수:", num_classes)
    
    # 테스트 데이터셋 생성
    test_dataset = tf.data.Dataset.from_generator(
        lambda: load_coco_dataset(args.test_annotations, args.test_images, id_mapping),
        output_types=(tf.string, tf.uint8),
        output_shapes=((), (None, None))
    )
    
    AUTOTUNE = tf.data.AUTOTUNE
    test_dataset = (test_dataset
                    .map(parse_image_mask, num_parallel_calls=AUTOTUNE)
                    .map(resize_image_mask, num_parallel_calls=AUTOTUNE)
                    .batch(args.batch_size)
                    .prefetch(AUTOTUNE))
    
    # 모델 로드 (coral_deeplab 패키지를 사용)
    try:
        import coral_deeplab as cdl
        model = cdl.applications.CoralDeepLabV3Plus(input_shape=(513, 513, 3),
                                                    n_classes=num_classes)
    except ImportError:
        raise ImportError("CoralDeepLabV3 모델을 찾을 수 없습니다. coral_deeplab 패키지 설치 또는 모델 구현을 확인하세요.")
    
    # 저장된 체크포인트에서 모델 가중치 로드
    if not os.path.exists(args.model_checkpoint):
        raise FileNotFoundError("모델 체크포인트를 찾을 수 없습니다: {}".format(args.model_checkpoint))
    model.load_weights(args.model_checkpoint)
    print("모델 가중치를 로드했습니다:", args.model_checkpoint)
    
    # mIoU 계산을 위한 MeanIoU metric 초기화
    miou_metric = tf.keras.metrics.MeanIoU(num_classes=num_classes)
    
    # 추가 평가를 위한 혼동 행렬 초기화
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    batch_index = 0
    for batch in test_dataset:
        batch_index += 1
        print(f"Processing batch {batch_index}")
        images, masks = batch
        logits = model.predict(images)
        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        
        # mIoU metric 업데이트
        miou_metric.update_state(masks, preds)
        
        # 혼동 행렬 업데이트 (배치 내 각 이미지별로 계산)
        masks_np = masks.numpy()
        preds_np = preds.numpy()
        for i in range(masks_np.shape[0]):
            gt_flat = masks_np[i].flatten()
            pred_flat = preds_np[i].flatten()
            conf_matrix += tf.math.confusion_matrix(gt_flat, pred_flat, num_classes=num_classes).numpy()
    
    overall_miou = miou_metric.result().numpy()
    print("전체 mIoU: {:.4f}".format(overall_miou))
    
    # 혼동 행렬을 바탕으로 추가 평가 지표 계산
    # Pixel Accuracy 계산
    total_correct = np.trace(conf_matrix)
    total_pixels = np.sum(conf_matrix)
    pixel_accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0
    
    # 각 클래스별 IoU 및 Dice coefficient 계산
    iou_list = []
    dice_list = []
    for i in range(num_classes):
        intersection = conf_matrix[i, i]
        gt_sum = np.sum(conf_matrix[i, :])
        pred_sum = np.sum(conf_matrix[:, i])
        union = gt_sum + pred_sum - intersection
        iou = intersection / union if union > 0 else 0.0
        dice = (2 * intersection) / (gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else 0.0
        iou_list.append(iou)
        dice_list.append(dice)
        print(f"클래스 {i} - IoU: {iou:.4f}, Dice: {dice:.4f}")
    
    mean_iou = np.mean(iou_list)
    mean_dice = np.mean(dice_list)
    
    # Frequency Weighted IoU 계산
    freq_weighted_iou = 0.0
    for i in range(num_classes):
        gt_pixels = np.sum(conf_matrix[i, :])
        freq_weighted_iou += (gt_pixels * iou_list[i])
    freq_weighted_iou = freq_weighted_iou / total_pixels if total_pixels > 0 else 0.0
    
    print("\n추가 평가 지표:")
    print("Pixel Accuracy: {:.4f}".format(pixel_accuracy))
    print("Mean IoU: {:.4f}".format(mean_iou))
    print("Mean Dice Coefficient: {:.4f}".format(mean_dice))
    print("Frequency Weighted IoU: {:.4f}".format(freq_weighted_iou))
    
    # mAP 계산은 각 클래스별 예측 확률이 필요하므로, 별도의 처리가 필요합니다.
    # 이 부분은 데이터셋 및 모델 출력 구조에 따라 추가 구현이 필요합니다.
    print("테스트 결과 평가 완료.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="테스트 데이터셋에 대해 시멘틱 세그멘테이션 모델 평가 (mIoU, Pixel Accuracy, Dice 등)")
    parser.add_argument("--test_annotations", type=str, required=True, help="테스트용 COCO JSON 어노테이션 경로")
    parser.add_argument("--test_images", type=str, required=True, help="테스트 이미지 디렉토리 경로")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="로드할 모델 체크포인트 경로")
    parser.add_argument("--batch_size", type=int, default=4, help="배치 크기")
    args = parser.parse_args()
    
    main(args)
