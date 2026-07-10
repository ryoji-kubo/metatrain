# Reuse PET's trainer to keep the same data preprocessing, augmentation, scaling,
# losses, batching, metric logging, and checkpointing.
from metatrain.pet.trainer import Trainer


__all__ = ["Trainer"]
