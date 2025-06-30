from __future__ import annotations

"""Sensor embedding layers for *coral-deeplab*.

This is a thin re-export of `BaseSensor` which lives in `fusion.py` so that
users can write `from coral_deeplab.sensors import BaseSensor` similarly to the
original PyTorch project layout (`models/sensor/base_sensor.py`).
"""

from .fusion import BaseSensor  # re-export

__all__ = ["BaseSensor"]
