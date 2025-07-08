"""Training helpers originally located in PyTorch version.

For the TensorFlow/Keras rewrite we only need a MeanIoU metric that
converts model logits to class indices (arg-max) before delegating to
`tf.keras.metrics.MeanIoU`.
"""

from __future__ import annotations

from typing import Any, Dict

import tensorflow as tf

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
