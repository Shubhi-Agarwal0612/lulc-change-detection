from .biunet import BiUNet
from .snunet_ecam import SNUNet_ECAM
from .siamunet_diff import SiamUNet_diff


MODELS = {
    "biunet": BiUNet,
    "snunet-ecam": SNUNet_ECAM,
    "siamunet-diff": SiamUNet_diff,
}


def build_model(name: str, num_bands: int = 3, num_classes: int = 2):
    """Build a change detection model by name.

    Args:
        name: One of 'biunet', 'snunet-ecam', 'siamunet-diff'.
        num_bands: Number of spectral bands per image (3 = RGB).
        num_classes: Number of output classes (2 for binary change).

    Returns:
        nn.Module ready for training.
    """
    key = name.lower()
    if key not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODELS.keys())}")

    if key == "biunet":
        return MODELS[key](in_channels=num_bands * 2, num_classes=num_classes)
    else:
        # Siamese models take separate T1, T2 inputs
        return MODELS[key](in_ch=num_bands, out_ch=num_classes)
