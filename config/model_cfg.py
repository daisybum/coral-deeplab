"""Configuration module originally written for the PyTorch training script.

Re-implemented here so that existing import lines like

    from config.model_cfg import *

continue to work in the TensorFlow port.  All data structures are implemented
with standard `dataclasses` / `Enum` so they are framework-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional

################################################################################
# Enumerations
################################################################################


class MODELTYPE(str, Enum):
    """High-level model family."""

    DEEPLABV3 = "deeplabv3"
    MASKRCNN = "maskrcnn"


class DEEPLABTYPE(str, Enum):
    RESNET50 = "resnet50"
    RESNET50V2 = "resnet50v2"


class MASKRCNNTYPE(str, Enum):
    RESNET50V2 = "resnet50v2"


################################################################################
# Dataclasses – mirror the original *Entity* objects
################################################################################


@dataclass
class Deeplabv3Entity:
    type: DEEPLABTYPE
    num_classes: int
    use_cbam: bool = False
    params: Dict[str, Any] | None = None
    load_from: Optional[str] = None


@dataclass
class MaskRcnnEntity:
    type: MASKRCNNTYPE
    params: Dict[str, Any]


@dataclass
class DataEntity:
    root_path: str
    image_base_path: str
    train_anno_path: str
    valid_anno_path: str
    test_anno_path: str


@dataclass
class DataloaderEntity:
    bathc_size: int
    num_worker: int = 0
    shuffle: bool = False


@dataclass
class TrainEntity:
    accum_step: int = 1
    num_epoch: int = 10
    log_step: int = 20


################################################################################
# Static configuration (mirrors the pasted snippet)
################################################################################

classes: int = 5
DEVICE: str = "cuda:0"  # For compatibility – TensorFlow will map to /GPU:0
MODEL_TYPE: MODELTYPE = MODELTYPE.DEEPLABV3
SENSOR_RATIO: float = 0.3
RESIZE_SCALE: int = 65  # Backbone feature map width/height

DATA_CFG = DataEntity(
    root_path="/workspace/data",
    image_base_path="images/",
    train_anno_path="COCO/train_without_street.json",
    valid_anno_path="COCO/valid_without_street.json",
    test_anno_path="COCO/test_without_street.json",
)

TRAIN_DATALOADER_CFG = DataloaderEntity(
    bathc_size=16,
    num_worker=0,
    shuffle=True,
)

VALID_DATALOADER_CFG = DataloaderEntity(
    bathc_size=16,
    num_worker=0,
    shuffle=False,
)

TEST_DATALOADER_CFG = DataloaderEntity(
    bathc_size=6,
    num_worker=0,
    shuffle=False,
)

TRAIN_CFG = TrainEntity(
    accum_step=1,
    num_epoch=10,
    log_step=20,
)

# Simple torchvision-style pipeline placeholder (dict list)
TRAIN_PIPE: List[Dict[str, Any]] = [
    {"type": "ToPILImage", "params": {}},
    {"type": "Resize", "params": {"size": (513, 513)}},
    {"type": "ToTensor", "params": {}},
]

VALID_PIPE: List[Dict[str, Any]] = [
    {"type": "ToPILImage", "params": {}},
    {"type": "Resize", "params": {"size": (513, 513)}},
    {"type": "ToTensor", "params": {}},
]

TEST_PIPE: List[Dict[str, Any]] = [
    {"type": "ToPILImage", "params": {}},
    {"type": "Resize", "params": {"size": (513, 513)}},
    {"type": "ToTensor", "params": {}},
]

MODEL_CFG: Dict[MODELTYPE, Any] = {
    MODELTYPE.DEEPLABV3: Deeplabv3Entity(
        type=DEEPLABTYPE.RESNET50,
        num_classes=1,
        use_cbam=True,
        params={"pretrained": True},
        load_from="resource/cls_weights.pth",
    ),
    MODELTYPE.MASKRCNN: MaskRcnnEntity(
        type=MASKRCNNTYPE.RESNET50V2,
        params={"num_classes": classes},
    ),
}
