"""TensorFlow training / validation script replacing the original PyTorch *main.py*.

This version trains *CoralDeepLab* models end-to-end using Keras.  It relies on
`coral_deeplab.utils.img_cls_dataset_tf.ImgClsDatasetTF` for data loading and
`coral_deeplab.applications` for model definitions.

Example
-------
$ python main_tf.py \
    --train_annotations data/train.json \
    --train_images    data/train_imgs \
    --val_annotations data/val.json   \
    --val_images      data/val_imgs   \
    --model deeplabv3plus --n_classes 6 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List

import tensorflow as tf
from tensorflow.keras import optimizers, losses, callbacks

from coral_deeplab.applications import CoralDeepLabV3, CoralDeepLabV3Plus
from config import model_cfg as cfg_mod
from coral_deeplab.utils.img_cls_dataset import ImgClsDatasetTF
from coral_deeplab.utils.transforms import get_transforms
from train import MeanIoUWithArgmax  # reuse metric defined in train.py

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CoralDeepLab models (TensorFlow)")
    parser.add_argument("--train_annotations", default=os.path.join(cfg_mod.DATA_CFG.root_path, cfg_mod.DATA_CFG.train_anno_path), type=str)
    parser.add_argument("--train_images",      default=os.path.join(cfg_mod.DATA_CFG.root_path, cfg_mod.DATA_CFG.image_base_path), type=str)
    parser.add_argument("--val_annotations",   default=os.path.join(cfg_mod.DATA_CFG.root_path, cfg_mod.DATA_CFG.valid_anno_path), type=str)
    parser.add_argument("--val_images",        default=os.path.join(cfg_mod.DATA_CFG.root_path, cfg_mod.DATA_CFG.image_base_path), type=str)
    parser.add_argument("--batch_size",        default=cfg_mod.TRAIN_DATALOADER_CFG.bathc_size,  type=int)
    parser.add_argument("--epochs",            default=cfg_mod.TRAIN_CFG.num_epoch, type=int)
    parser.add_argument("--model",             default="deeplabv3", choices=["deeplabv3", "deeplabv3plus"])
    parser.add_argument("--n_classes",         default=cfg_mod.classes, type=int)
    parser.add_argument("--lr",                default=1e-4, type=float)
    parser.add_argument("--output",            default="checkpoints_tf", type=str)
    # Optional transforms pipelines as JSON string (list of dicts)
    parser.add_argument("--train_pipeline",    default=json.dumps(cfg_mod.TRAIN_PIPE), type=str)
    parser.add_argument("--val_pipeline",      default=json.dumps(cfg_mod.VALID_PIPE), type=str)
    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------


def build_model(model_name: str, input_shape=(513, 513, 3), n_classes: int = 6):
    if model_name == "deeplabv3":
        return CoralDeepLabV3(input_shape=input_shape, n_classes=n_classes)
    elif model_name == "deeplabv3plus":
        return CoralDeepLabV3Plus(input_shape=input_shape, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model {model_name}")


# --------------------------------------------------------------------------------------
# Main training logic
# --------------------------------------------------------------------------------------


def main(cfg: argparse.Namespace):
    os.makedirs(cfg.output, exist_ok=True)

    # ------------------------------------------------------------------
    # Load COCO-style JSON annotations into raw_data dicts
    # ------------------------------------------------------------------
    def load_json(path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)

    train_raw = load_json(cfg.train_annotations)
    val_raw   = load_json(cfg.val_annotations)

    # Determine number of classes from dataset category ids
    dataset_n_classes = max(cat["id"] for cat in train_raw["categories"]) + 1
    n_classes = dataset_n_classes

    # Path configuration for dataset – taken directly from config
    data_cfg = cfg_mod.DATA_CFG

    # ------------------------------------------------------------------
    # Albumentations / tf.image transforms
    # ------------------------------------------------------------------
    train_pipe_cfg: List[Dict[str, Any]] = json.loads(cfg.train_pipeline) if cfg.train_pipeline else cfg_mod.TRAIN_PIPE
    val_pipe_cfg:   List[Dict[str, Any]] = json.loads(cfg.val_pipeline) if cfg.val_pipeline else cfg_mod.VALID_PIPE

    # TensorFlow-friendly image transforms (apply later within tf.data pipeline)
    train_img_tfms = get_transforms(train_pipe_cfg) if train_pipe_cfg else None
    val_img_tfms   = get_transforms(val_pipe_cfg)   if val_pipe_cfg   else None

    # ------------------------------------------------------------------
    # Determine resize target from Resize transform (shared for img & mask)
    # ------------------------------------------------------------------
    resize_size = None
    for t in train_pipe_cfg:
        if t["type"] == "Resize" and "params" in t:
            sz = t["params"].get("size")
            if sz is not None:
                resize_size = (sz, sz) if isinstance(sz, int) else tuple(sz)
            break

    STRIDE = 16  # Deeplab default output stride

    def _resize_mask_fn(m):
        if resize_size is None:
            return m
        # Downsample mask to logits resolution (ceil resize_size / stride)
        out_h = (resize_size[0] + STRIDE - 1) // STRIDE
        out_w = (resize_size[1] + STRIDE - 1) // STRIDE
        m = tf.expand_dims(m, -1)
        m = tf.image.resize(m, (out_h, out_w), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        m = tf.squeeze(m, -1)
        return tf.cast(m, tf.int32)


    # ------------------------------------------------------------------
    # Build tf.data.Dataset pipelines (Albumentations disabled for now)
    # ------------------------------------------------------------------
    train_ds_gen = ImgClsDatasetTF(data_cfg, train_raw, transforms=None)
    val_ds_gen   = ImgClsDatasetTF(data_cfg, val_raw,   transforms=None)

    train_ds = train_ds_gen.as_dataset(batch_size=cfg.batch_size, shuffle=True)
    val_ds   = val_ds_gen.as_dataset(batch_size=cfg.batch_size, shuffle=False)

    # Optional TensorFlow image transforms + drop unused fields (sensors, annots)
    # Apply image transforms and ensure mask is resized consistently
    train_ds = train_ds.map(
        lambda img, mask, sensors, ann: (
            (
                train_img_tfms(img) if train_img_tfms else img,
                sensors,
            ),
            _resize_mask_fn(mask),
        )
    )

    val_ds = val_ds.map(
        lambda img, mask, sensors, ann: (
            (
                val_img_tfms(img) if val_img_tfms else img,
                sensors,
            ),
            _resize_mask_fn(mask),
        )
    )

    # ------------------------------------------------------------------
    # Model / Loss / Optimizer
    # ------------------------------------------------------------------
    # Infer input shape from Resize transform (if present) so model and data align
    resize_size = None
    for t in train_pipe_cfg:
        if t["type"] == "Resize" and "params" in t:
            sz = t["params"].get("size")
            if sz is not None:
                if isinstance(sz, int):
                    resize_size = (sz, sz)
                else:
                    resize_size = tuple(sz)
            break

    input_shape = resize_size + (3,) if resize_size else (513, 513, 3)

    model = build_model(cfg.model, input_shape=input_shape, n_classes=n_classes)
    loss  = losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizers.Adam(cfg.lr),
        loss=loss,
        metrics=[MeanIoUWithArgmax(num_classes=n_classes)],
    )

    ckpt_cb = callbacks.ModelCheckpoint(
        filepath=os.path.join(cfg.output, "epoch_{epoch:02d}.keras"),
        save_weights_only=True,
        save_best_only=False,
    )

    lr_cb = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=[ckpt_cb, lr_cb],
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
