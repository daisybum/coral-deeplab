from __future__ import annotations

"""CBAM (Convolutional Block Attention Module) layers implemented in TensorFlow / Keras

This file is modeled after the PyTorch implementation located in
``deeplab-sensor-fusion/models/cbam/cbam.py`` and ports it to TensorFlow so it can
be plugged into the *coral-deeplab* code-base.

The module exposes three public symbols:
    - `conv_block`:   small Conv-BN-ReLU helper (identical semantics to the PyTorch helper)
    - `Cbam`:         the complete CBAM block (Channel + Spatial attention)
    - `SAM`, `CAM`:   the spatial and channel sub-blocks (exported mainly for completeness)

Example
-------
>>> import tensorflow as tf
>>> from coral_deeplab.attention import Cbam
>>> x = tf.random.normal([1, 64, 64, 256])  # BCHW->TensorFlow uses NHWC layout
>>> cbam = Cbam(channels=256, reduction_ratio=8)
>>> y = cbam(x)
>>> assert y.shape == x.shape
"""

import tensorflow as tf
from tensorflow.keras import layers


################################################################################
# Utility
################################################################################

def conv_block(
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    activation: str | None = None,
    name: str | None = None,
) -> tf.keras.Sequential:
    """3×3 Conv + BN + ReLU block used repeatedly inside the EESP-like heads.

    Parameters
    ----------
    filters : int
        Number of output feature maps.
    kernel_size : int, default=3
        Convolution kernel size.
    strides : int, default=1
        Convolution strides.
    activation : str | None, default=None ("relu")
        Optional activation to append (if not supplied the caller can add it).
    name : str | None
        Base name for the layers.
    """

    if activation is None:
        activation = "relu"

    block = tf.keras.Sequential(name=name)
    block.add(
        layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding="same",
            use_bias=False,
            name=f"{name}_conv" if name else None,
        )
    )
    block.add(layers.BatchNormalization(name=f"{name}_bn" if name else None))
    block.add(layers.ReLU(name=f"{name}_relu" if name else None))
    return block


################################################################################
# Channel Attention Module
################################################################################

class CAM(layers.Layer):
    """Channel attention as defined in the CBAM paper."""

    def __init__(self, channels: int, reduction_ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        # shared MLP – implemented with Dense layers that will be re-used for
        # both average- and max-pooled descriptors.
        self.fc1 = layers.Dense(
            channels // reduction_ratio,
            activation="relu",
            use_bias=True,
            name="mlp_fc1",
        )
        self.fc2 = layers.Dense(
            channels,
            activation=None,
            use_bias=True,
            name="mlp_fc2",
        )
        self.reshape = layers.Reshape((1, 1, channels))

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:  # NHWC
        # Global avg-/max-pooling → [B, C]
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2])
        max_pool = tf.reduce_max(inputs, axis=[1, 2])

        # Shared MLP for both descriptors
        avg_out = self.fc2(self.fc1(avg_pool))
        max_out = self.fc2(self.fc1(max_pool))
        scale = tf.nn.sigmoid(avg_out + max_out)  # [B, C]
        scale = self.reshape(scale)               # [B, 1, 1, C]
        return inputs * scale


################################################################################
# Spatial Attention Module
################################################################################

class SAM(layers.Layer):
    """Spatial attention as defined in the CBAM paper."""

    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            use_bias=False,
            name="sam_conv",
        )

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:  # NHWC
        # Channel-wise statistics
        avg_pool = tf.reduce_mean(inputs, axis=3, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=3, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=3)  # [B, H, W, 2]
        scale = tf.nn.sigmoid(self.conv(concat))          # [B, H, W, 1]
        return inputs * scale


################################################################################
# CBAM wrapper
################################################################################

class Cbam(layers.Layer):
    """Full Convolutional Block Attention Module (Channel → Spatial).

    The output is the attention-enhanced feature *added* to the residual input
    (this matches the PyTorch implementation used in the sensor-fusion repo).
    """

    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        kernel_size: int = 7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channel_att = CAM(channels, reduction_ratio, name="cam")
        self.spatial_att = SAM(kernel_size, name="sam")

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:  # NHWC
        x = self.channel_att(inputs)
        x = self.spatial_att(x)
        return x + inputs

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "channels": self.channel_att.channels,
                "reduction_ratio": self.channel_att.reduction_ratio,
                "kernel_size": self.spatial_att.kernel_size,
            }
        )
        return cfg
