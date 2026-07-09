from .base_dataset import RemoteSensingDetectionDataset
from .dior import DIORDataset
from .dota import DOTADataset
from .fair1m import FAIR1MDataset

DATASET_REGISTRY = {
    "DIOR": DIORDataset,
    "DOTA": DOTADataset,
    "FAIR1M": FAIR1MDataset,
}


def build_dataset(name: str, **kwargs):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](**kwargs)


__all__ = [
    "RemoteSensingDetectionDataset",
    "DIORDataset",
    "DOTADataset",
    "FAIR1MDataset",
    "DATASET_REGISTRY",
    "build_dataset",
]
