from .model import StructureTransformerModel
from .trainer import Trainer


__model__ = StructureTransformerModel
__trainer__ = Trainer
__capabilities__ = {
    "supported_devices": __model__.__supported_devices__,
    "supported_dtypes": __model__.__supported_dtypes__,
}

__authors__ = [
    ("Ryoji / Codex adapter", ""),
]

__maintainers__ = [
    ("Ryoji", ""),
]
