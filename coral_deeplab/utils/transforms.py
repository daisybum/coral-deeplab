"""TensorFlow helpers mirroring `utils/transforms.py` from the PyTorch code-base.

The original implementation relied on *torchvision* for image transforms.  This
version re-implements a minimal subset using **TensorFlow / tf.image** so that
the rest of the training pipeline can remain framework-agnostic.

Currently supported transform names (``config["type"]``):
* ``Resize`` – params: ``size`` (tuple | int), ``method`` (str, optional)
* ``RandomHorizontalFlip`` – params: ``prob`` (float, default=0.5)
* ``RandomVerticalFlip`` – params: ``prob`` (float, default=0.5)

You can easily extend the ``_TRANSFORMS`` mapping below to add more ops.

All helper functions (``get_mask_from_polygon`` & ``resize_seg_ptrs``) keep the
original NumPy / OpenCV behaviour to remain compatible with downstream code.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Callable, Dict, Any
import cv2
import numpy as np
import tensorflow as tf

__all__ = [
    "get_transforms",
    "get_mask_from_polygon",
    "resize_seg_ptrs",
]

# -----------------------------------------------------------------------------
# Transform factory
# -----------------------------------------------------------------------------

TransformFn = Callable[[tf.Tensor], tf.Tensor]


def _resize_factory(params: Dict[str, Any]) -> TransformFn:
    size = params.get("size")
    if size is None:
        raise ValueError("Resize transform requires 'size' param (int or tuple)")
    if isinstance(size, int):
        size = (size, size)
    method_str = params.get("method", "bilinear").lower()
    method = {
        "bilinear": tf.image.ResizeMethod.BILINEAR,
        "nearest": tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        "bicubic": tf.image.ResizeMethod.BICUBIC,
    }.get(method_str, tf.image.ResizeMethod.BILINEAR)

    def _resize(img: tf.Tensor) -> tf.Tensor:  # img: float32 [0,1] or uint8
        return tf.image.resize(img, size, method=method)

    return _resize


def _rand_hflip_factory(params: Dict[str, Any]) -> TransformFn:
    prob = params.get("prob", 0.5)

    def _flip(img: tf.Tensor) -> tf.Tensor:
        return tf.image.random_flip_left_right(img, seed=None) if tf.random.uniform(()) < prob else img

    return _flip


def _rand_vflip_factory(params: Dict[str, Any]) -> TransformFn:
    prob = params.get("prob", 0.5)

    def _flip(img: tf.Tensor) -> tf.Tensor:
        return tf.image.random_flip_up_down(img, seed=None) if tf.random.uniform(()) < prob else img

    return _flip


_TRANSFORMS: Dict[str, Callable[[Dict[str, Any]], TransformFn]] = {
    "Resize": _resize_factory,
    "RandomHorizontalFlip": _rand_hflip_factory,
    "RandomVerticalFlip": _rand_vflip_factory,
    "ToPILImage": lambda params: (lambda img: img),  # no-op for TF pipeline
    "ToTensor": lambda params: (lambda img: img),  # images are converted to tensors later
}


def get_transforms(cfg: Dict[str, Any] | List[Dict[str, Any]]) -> Callable[[tf.Tensor], tf.Tensor]:
    """Return a composite transform function built from cfg.

    Parameters
    ----------
    cfg : dict | list[dict]
        Either a single transform config *or* a list of configs.  Each dict must
        have keys ``type`` and ``params``.
    """
    if isinstance(cfg, dict):
        cfg = [cfg]

    fns: List[TransformFn] = []
    for tconf in cfg:
        tname = tconf["type"]
        params = tconf.get("params", {})
        if tname not in _TRANSFORMS:
            raise ValueError(f"{tname} is not supported by transforms_tf.py")
        fns.append(_TRANSFORMS[tname](params))

    def _compose(img: tf.Tensor) -> tf.Tensor:
        for fn in fns:
            img = fn(img)
        return img

    return _compose

# -----------------------------------------------------------------------------
# Polygon helpers (identical logic to original version)
# -----------------------------------------------------------------------------

def get_mask_from_polygon(img_size: Tuple[int, int], num_classes: int, annotations) -> np.ndarray:
    """Rasterise polygon annotations into an integer mask (H×W)."""
    scale_cnt = len(img_size)
    if scale_cnt not in (1, 2):
        raise ValueError(
            f"Resize ptr error. resize_scale's shape is only 1 or 2. {scale_cnt} is not supported."
        )
    mask = np.zeros((img_size[0], img_size[-1]), dtype=np.uint8)
    for anno in annotations:
        if "segmentation" in anno:
            cat_id = anno["category_id"]
            for seg in anno["segmentation"]:
                polygon = np.asarray(seg, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [polygon], cat_id)
    return mask


def resize_seg_ptrs(
    annotations: List[dict],
    original_size: Tuple[int, int],  # (h, w)
    resize_scale: Optional[Tuple[int, int] | Tuple[int]],  # (h, w) or (s,)
) -> Tuple[float, float]:
    """Scale polygon points to match a resized image/feature map size."""
    if resize_scale is None:
        return 1.0, 1.0

    scale_cnt = len(resize_scale)
    if scale_cnt not in (1, 2):
        raise ValueError(
            f"Resize ptr error. resize_scale's shape is only 1 or 2. {scale_cnt} is not supported."
        )

    y_ratio = resize_scale[0] / original_size[0]
    x_ratio = resize_scale[-1] / original_size[1]

    for anno in annotations:
        if not anno.get("is_scale", False) and "segmentation" in anno:
            for seg in anno["segmentation"]:
                for ptr in range(0, len(seg), 2):
                    seg[ptr] *= x_ratio
                    seg[ptr + 1] *= y_ratio
            anno["is_rescale"] = True

    return y_ratio, x_ratio
