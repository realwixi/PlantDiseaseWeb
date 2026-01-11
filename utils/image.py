from typing import Tuple
import numpy as np

try:
    from tensorflow.keras.applications.efficientnet import preprocess_input
except Exception:
    preprocess_input = None


def preprocess_rgb_numpy(x: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess an RGB numpy image for EfficientNetB0.
    Assumes input is already resized to `size` and dtype float32.
    """
    if preprocess_input is not None:
        return preprocess_input(x)
    # Fallback normalization if TF isn't available yet
    return x / 255.0
