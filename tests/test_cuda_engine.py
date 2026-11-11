import sys
import pytest


def test_cuda_engine_import_without_torch(monkeypatch):
    """Test importing parallax.cuda_engine when torch is not available."""
    # Ensure torch cannot be imported
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", None)
    
    # Remove parallax.cuda_engine if cached
    sys.modules.pop("parallax.cuda_engine", None)
    
    import parallax.cuda_engine as ce
    assert ce.is_cuda_available() is False
