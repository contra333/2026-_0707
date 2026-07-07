"""Model architectures and factories."""

from .factory import make_model
from .simple_cnn import SimpleCNN

__all__ = ["SimpleCNN", "make_model"]
