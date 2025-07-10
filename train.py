"""Training helpers originally located in PyTorch version.

For the TensorFlow/Keras rewrite we only need a MeanIoU metric that
converts model logits to class indices (arg-max) before delegating to
`tf.keras.metrics.MeanIoU`.
"""

from __future__ import annotations

from typing import Any, Dict

import tensorflow as tf

# ------------------------------------------------------------------
# GPU 메모리 제한 – 최대 10GB (10240MB)
# ------------------------------------------------------------------
# *train.py* 는 모델 학습/평가 스크립트(main.py)에서 import 되므로,
# 이 블록을 모듈 로드 시점에 실행하면 자연스럽게 GPU 메모리 사용량을 제한할 수 있다.

try:
    _gpus = tf.config.list_physical_devices("GPU")
    if _gpus:
        tf.config.experimental.set_virtual_device_configuration(
            _gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)],
        )
        # 메모리 성능 최적화: 필요 시 growth 옵션 대신 정적 할당 사용 중.
        print("[INFO] GPU 메모리 제한: 8GB (8192MB) 적용 완료")
except RuntimeError as _e:
    # set_virtual_device_configuration() 은 물리적 장치를 초기화하기 전에만 호출 가능
    print(f"[WARN] GPU 메모리 제한 설정 실패: {_e}")

__all__ = ["MeanIoUWithArgmax"]


class MeanIoUWithArgmax(tf.keras.metrics.MeanIoU):
    """Mean Intersection-over-Union metric that accepts logits.

    Keras' built-in `MeanIoU` expects *integer* class predictions.
    During training, however, models typically output raw logits
    or probabilities.  ``MeanIoUWithArgmax`` first applies
    ``tf.argmax`` to convert logits/probabilities to integer class
    IDs and then forwards the result to the regular ``MeanIoU``
    implementation.
    """

    def __init__(self, num_classes: int, name: str = "mean_iou_with_argmax", **kwargs: Any):
        super().__init__(num_classes=num_classes, name=name, **kwargs)
        self._num_classes = num_classes

    # ------------------------------------------------------------------
    # Keras API overrides
    # ------------------------------------------------------------------

    def update_state(self, y_true, y_pred, sample_weight=None):  # type: ignore[override]
        # Convert logits/probabilities to integer predictions.
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred_classes, sample_weight=sample_weight)

    def get_config(self) -> Dict[str, Any]:  # noqa: D401
        """Return the metric configuration for serialization."""
        cfg = super().get_config()
        cfg.update({"num_classes": self._num_classes})
        return cfg
