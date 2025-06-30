from __future__ import annotations

"""Model factory utilities similar to *deeplab-sensor-fusion/models/functional.py*.

Only TensorFlow/Keras implementations relevant for **coral-deeplab** are
included (currently DeepLabV3 with optional CBAM + Sensor fusion). Mask-RCNN is
not supported in this repo and will raise ``NotImplementedError``.
"""

from enum import Enum
from typing import NamedTuple, Tuple

import tensorflow as tf

from .attention import Cbam
from .fusion import SensorVisionFusion
from ._blocks import deeplab_aspp_module, deeplabv3_decoder
from ._encoders import mobilenetv2

###############################################################################
# Enums & Config containers
###############################################################################

class MODELTYPE(str, Enum):
    DEEPLABV3 = "deeplabv3"
    MASKRCNN = "maskrcnn"  # not implemented; kept for API parity


class DeeplabConfig(NamedTuple):
    num_classes: int = 6
    input_shape: Tuple[int, int, int] = (513, 513, 3)
    alpha: float = 1.0  # MobileNetV2 width multiplier
    use_cbam: bool = True
    use_fusion: bool = True
    sensor_ratio: float = 0.3
    reduction_ratio: int = 8  # CBAM reduction ratio
    num_sensor_values: int = 6


###############################################################################
# Builders
###############################################################################

def _build_deeplab_model(cfg: DeeplabConfig) -> tf.keras.Model:
    """Constructs a DeepLabV3 (+CBAM + SensorFusion) Keras model."""

    # Inputs
    image_in = tf.keras.Input(shape=cfg.input_shape, name="image")
    sensor_in = tf.keras.Input(shape=(cfg.num_sensor_values,), name="sensors")

    # Encoder backbone (MobileNetV2 variant used in coral-deeplab)
    x = mobilenetv2(image_in, alpha=cfg.alpha)

    # Optional CBAM on encoder output
    if cfg.use_cbam:
        x = Cbam(channels=x.shape[-1], reduction_ratio=cfg.reduction_ratio, name="cbam_att")(x)

    # Optional sensor fusion before ASPP
    if cfg.use_fusion:
        x = SensorVisionFusion(channels=x.shape[-1], sensor_ratio=cfg.sensor_ratio, name="sensor_fusion")(
            [x, sensor_in]
        )

    # ASPP & decoder head
    aspp_out = deeplab_aspp_module(x)
    logits = deeplabv3_decoder(aspp_out, cfg.num_classes)

    model = tf.keras.Model(inputs=[image_in, sensor_in], outputs=logits, name="CoralDeepLabV3Sensor")
    return model


###############################################################################
# Public API
###############################################################################

def get_model(model_type: MODELTYPE, config: DeeplabConfig | None = None):
    """Factory that mimics the original PyTorch `get_model` interface."""

    if model_type == MODELTYPE.DEEPLABV3:
        if config is None:
            config = DeeplabConfig()  # use defaults
        return _build_deeplab_model(config)

    elif model_type == MODELTYPE.MASKRCNN:
        raise NotImplementedError("MaskRCNN is not supported in the TensorFlow port.")

    else:
        raise ValueError(f"Unknown MODELTYPE: {model_type}")
