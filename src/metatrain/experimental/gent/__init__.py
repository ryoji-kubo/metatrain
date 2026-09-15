from .model import GenTModel
from .trainer import Trainer


__model__ = GenTModel
__trainer__ = Trainer
__capabilities__ = {
    "supported_devices": __model__.__supported_devices__,
    "supported_dtypes": __model__.__supported_dtypes__,
}
__authors__ = [
    ("Shao Yanming", ""),
    ("Xavier Bresson", ""),
    ("Ryoji / Codex adapter", ""),
]
__maintainers__ = [("Ryoji", "")]
