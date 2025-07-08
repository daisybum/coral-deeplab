"""TensorFlow dataset equivalent of ImgClsDataset for *coral-deeplab*.

This module provides `ImgClsDatasetTF`, a thin wrapper that mirrors the original
PyTorch‐based `ImgClsDataset` API but outputs data compatible with TensorFlow
and `tf.data.Dataset` pipelines.

Core differences from the PyTorch version
-----------------------------------------
* Images are loaded via OpenCV **BGR → RGB** conversion and returned as
  `tf.float32` tensors in the **[0, 1]** range.
* Masks are returned as `tf.int32` tensors (H×W) where each pixel stores the
  *category id* (0 … N-1).
* Albumentations transforms (``albumentations.Compose``) are supported – they
  will be applied to **numpy** images/masks first and converted to tensors
  afterwards.
* The generator interface (`as_dataset`) allows easy conversion into
  `tf.data.Dataset`.

Example
-------
>>> dataset = ImgClsDatasetTF(cfg, raw_data, transforms=alb_transforms)
>>> tf_ds  = dataset.as_dataset(batch_size=8, shuffle=True)

The dataset yields `(image, mask, sensors, annotations)` just like the original
version.  `annotations` is a `dict` with additional `image_path` added.
"""

from __future__ import annotations

from typing import Optional, Any, Iterable, Tuple, List, Dict
import os

import cv2
import json
import numpy as np
import tensorflow as tf

try:
    from albumentations import Compose
except ImportError:  # graceful fallback when Albumentations is not installed
    Compose = None  # type: ignore

# -----------------------------------------------------------------------------
# Helper Utilities
# -----------------------------------------------------------------------------

def get_mask_from_polygon(img_shape: Tuple[int, int], num_categories: int, annotations: List[Dict]) -> np.ndarray:
    """Rasterise polygon annotations to a H×W integer mask.

    This is a lightweight re-implementation that avoids a heavy COCO dependency.
    Each annotation must contain *category_id* and *segmentation* (list of
    polygons).  The returned mask has shape (H, W) and integer dtype where
    pixel values equal the category id.  Background == 0.
    """
    height, width = img_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    for ann in annotations:
        cat_id = ann.get("category_id", 0)
        segm = ann.get("segmentation", [])
        if not isinstance(segm, list):
            # Unsupported RLE segmentation – skip
            continue
        for poly in segm:
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], cat_id)
    return mask


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------

class ImgClsDatasetTF:
    """TensorFlow port of *ImgClsDataset* (sensor-aware image classification).

    Parameters
    ----------
    config : Any
        A simple config object that must expose ``root_path`` and
        ``image_base_path`` attributes.
    raw_data : dict
        Dictionary with COCO-style keys: ``images``, ``annotations``,
        ``categories``.
    transforms : albumentations.Compose | None
        Optional Albumentations augmentations applied on *numpy* images/masks.
    """

    def __init__(
        self,
        config: Any,
        raw_data: Dict[str, Any],
        transforms: Optional["Compose"] = None,
    ) -> None:
        self.config = config
        self.images: List[Dict] = list(raw_data["images"])
        self.annotations: List[Dict] = list(raw_data["annotations"])
        self.categories: List[Dict] = list(raw_data["categories"])
        self.transforms = transforms

        # Build mapping image_id -> annotation indices
        self.ia_map: Dict[int, List[int]] = {}
        for idx, ann in enumerate(self.annotations):
            img_id = ann["image_id"]
            self.ia_map.setdefault(img_id, []).append(idx)

        # Remove images without annotations
        self.images = [img for img in self.images if img["id"] in self.ia_map]

        # Deduce resize scale (if a Resize transform is present)
        self.resize_scale: Optional[Tuple[int, int]] = None
        if transforms is not None:
            from albumentations import Resize  # type: ignore
            for t in transforms.transforms:  # type: ignore[attr-defined]
                if isinstance(t, Resize):
                    self.resize_scale = (t.height, t.width)  # type: ignore[attr-defined]
                    break

    # ---------------------------------------------------------------------
    # Pythonic helpers (len / getitem) – useful for debugging or numpy usage
    # ---------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):  # type: ignore[override]
        img_info = self.images[index]
        img_id = img_info["id"]
        img_fn = img_info["file_name"]
        img_path = os.path.join(self.config.root_path, self.config.image_base_path, img_fn)

        # OpenCV loads in BGR – convert to RGB and float32 [0,1]
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        annos = [self.annotations[i] for i in self.ia_map[img_id]]
        mask = get_mask_from_polygon(img.shape[:2], len(self.categories), annos)
        if self.resize_scale is not None:
            mask = cv2.resize(mask, self.resize_scale, interpolation=cv2.INTER_NEAREST)

        sensors_raw = img_info.get("sensor_info", {})

        # ------------------------------------------------------------------
        # Convert sensor information (dict or list[dict]) into numeric vector
        # ------------------------------------------------------------------
        def _sensor_to_vec(s):
            if not s:
                return np.zeros(6, dtype=np.float32)
            if isinstance(s, list):
                s = s[0] if len(s) > 0 else {}
            # Expected keys and order
            keys = [
                "objectTemp",  # object temperature
                "humi",        # humidity
                "pressure",    # pressure (hPa)
                "latitude",
                "longitude",
                "height",
            ]
            vec = [float(s.get(k, 0.0)) for k in keys]
            return np.asarray(vec, dtype=np.float32)

        sensor_vec = _sensor_to_vec(sensors_raw)

        # Albumentations expects HWC uint8 image
        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Convert to TensorFlow tensors
        img_tf = tf.convert_to_tensor(img, dtype=tf.float32) / 255.0  # [0,1]
        mask_tf = tf.convert_to_tensor(mask, dtype=tf.int32)
        sensors_tf = tf.convert_to_tensor(sensor_vec, dtype=tf.float32)

        # Add image_path info for compatibility (kept as python str)
        for a in annos:
            a["image_path"] = img_path

        ann_json = json.dumps(annos)
        return img_tf, mask_tf, sensors_tf, ann_json

    # ------------------------------------------------------------------
    # Public API – create a tf.data.Dataset from the generator
    # ------------------------------------------------------------------

    def as_dataset(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        buffer_size: int = 1000,
        drop_remainder: bool = False,
    ) -> tf.data.Dataset:
        """Return a batched `tf.data.Dataset`."""

        def gen():  # generator yielding python objects
            for idx in range(len(self)):
                yield self[idx]

        output_signature = (
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),  # image
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),       # mask
            tf.TensorSpec(shape=(6,), dtype=tf.float32),             # sensors vector
            tf.TensorSpec(shape=(), dtype=tf.string),                # annotations JSON
        )

        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        if shuffle:
            ds = ds.shuffle(buffer_size)
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        return ds


__all__ = [
    "ImgClsDatasetTF",
    "get_mask_from_polygon",
]
