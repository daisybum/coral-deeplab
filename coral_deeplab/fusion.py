from __future__ import annotations

"""Sensor–Vision fusion layers implemented in TensorFlow / Keras.

These layers are a direct port of the PyTorch implementation in
``deeplab-sensor-fusion/models/fusion/sensor_vision_fusion.py`` and its
corresponding `BaseSensor`.  They allow numerical (e.g. IMU, Lidar) sensor
vectors to be embedded and added to convolutional feature maps inside a
TensorFlow/Keras segmentation network such as *coral-deeplab*.
"""

import tensorflow as tf
from tensorflow.keras import layers

# Default constants – can be overridden at layer construction time
_DEFAULT_RESIZE_SCALE = 65  # 65×65 matches DeepLabV3-MNV3 output @ 520×520 input
_DEFAULT_SENSOR_RATIO = 0.3


class BaseSensor(layers.Layer):
    """Embeds a 1-D sensor vector into a 2-D feature map.

    The original PyTorch version trains a simple MLP → reshape → 1×1 Conv.
    Here we replicate that behaviour.

    Parameters
    ----------
    channels : int, default=2048
        Number of feature channels to output (should match vision feature).
    resize_scale : int, default=65
        Spatial width/height of the output feature map.
    """

    def __init__(self, channels: int = 2048, resize_scale: int = _DEFAULT_RESIZE_SCALE, **kwargs):
        super().__init__(**kwargs)
        self.resize_scale = resize_scale
        self.dense = layers.Dense(
            units=resize_scale * resize_scale,
            activation=None,
            name="sensor_dense",
        )
        self.bn1d = layers.BatchNormalization(name="sensor_dense_bn")

        self.conv = layers.Conv2D(
            filters=channels,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name="sensor_conv1x1",
        )
        self.bn2d = layers.BatchNormalization(name="sensor_conv1x1_bn")
        self.relu = layers.ReLU(name="sensor_relu")

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # inputs shape: [B, S]  (S == number of sensor values, here 6)
        x = self.dense(inputs)
        x = self.bn1d(x)
        x = self.relu(x)
        # reshape to NHWC
        x = tf.reshape(x, [-1, self.resize_scale, self.resize_scale, 1])
        x = self.conv(x)
        x = self.bn2d(x)
        x = self.relu(x)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"resize_scale": self.resize_scale})
        return cfg


class SensorVisionFusion(layers.Layer):
    """Adds sensor features to vision features with a scaling factor."""

    def __init__(
        self,
        channels: int = 2048,
        sensor_ratio: float = _DEFAULT_SENSOR_RATIO,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sensor_ratio = sensor_ratio
        self.sensor_net = BaseSensor(channels, name="base_sensor")

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor] | list[tf.Tensor], **kwargs) -> tf.Tensor:
        """Expects *(vision_features, sensor_vector)* as input."""
        vision_features, sensors = inputs
        sensor_features = self.sensor_net(sensors)
        return vision_features + (sensor_features * self.sensor_ratio)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"sensor_ratio": self.sensor_ratio})
        return cfg
