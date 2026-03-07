"""
Shared pytest configuration and fixtures.

pytest-asyncio is set to 'auto' mode so async test functions
don't need the @pytest.mark.asyncio decorator explicitly.
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent

# Only add test-neutral dirs globally. Service dirs (nlp_service, asr_service, etc.)
# each contain a 'main.py' and 'model.py' — adding all of them globally causes
# pytest to find the WRONG 'main' when tests run in combined mode.
# Each test file adds its own service dir at the top of the file.
GLOBAL_DIRS = [
    ROOT / "database",
    ROOT / "tests",
]
for d in GLOBAL_DIRS:
    p = str(d)
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark test as async (handled by pytest-asyncio)"
    )
    # Install the torch stub once at session start so all endpoint tests that
    # import torch-dependent service code (sentiment/nlp) can run without torch.
    # The stub is a no-op if real torch is already installed.
    _install_torch_stub_if_needed()
    # Install the transformers stub for BertTokenizer / BertModel
    _install_transformers_stub_if_needed()


def _install_transformers_stub_if_needed():
    """Install a minimal transformers stub (BertTokenizer, BertModel) if not installed."""
    if "transformers" in sys.modules:
        return

    import types
    from unittest.mock import MagicMock

    transformers_mod = types.ModuleType("transformers")
    # Use MagicMock so tokenizer(...) returns dict of MagicMocks with .to() support
    transformers_mod.BertTokenizer = MagicMock()
    transformers_mod.BertModel     = MagicMock()
    sys.modules["transformers"] = transformers_mod


def _install_torch_stub_if_needed():
    """Install a minimal torch stub if torch is not installed."""
    if "torch" in sys.modules:
        return  # real torch present — nothing to do

    import types

    class _Module:
        def __init__(self): pass
        def to(self, *a, **kw): return self
        def eval(self): return self
        def train(self, mode=True): return self

    class _Linear(_Module):
        def __init__(self, *a, **kw): pass

    class _Embedding(_Module):
        def __init__(self, *a, **kw):
            self.weight = type("W", (), {"data": None})()

    class _Dropout(_Module):
        def __init__(self, *a, **kw): pass

    class _LSTM(_Module):
        def __init__(self, *a, **kw): pass

    nn_mod = types.ModuleType("torch.nn")
    nn_mod.Module    = _Module
    nn_mod.Linear    = _Linear
    nn_mod.Embedding = _Embedding
    nn_mod.Dropout   = _Dropout
    nn_mod.LSTM      = _LSTM
    nn_mod.CrossEntropyLoss = _Module
    nn_mod.init = types.SimpleNamespace(xavier_uniform_=lambda t: t)

    nn_functional = types.ModuleType("torch.nn.functional")
    nn_functional.softmax = lambda x, dim=None: x
    nn_functional.relu    = lambda x: x
    nn_functional.conv1d  = lambda *a, **kw: None   # needed by noisereduce.torchgate
    nn_functional.conv2d  = lambda *a, **kw: None
    nn_functional.pad     = lambda *a, **kw: None
    nn_functional.unfold  = lambda *a, **kw: None

    torch_mod = types.ModuleType("torch")
    torch_mod.__path__ = []   # makes it a package so submodule imports work
    torch_mod.__package__ = "torch"
    torch_mod.nn            = nn_mod
    torch_mod.device        = lambda s: s
    torch_mod.cuda          = types.SimpleNamespace(is_available=lambda: False)
    torch_mod.load          = lambda *a, **kw: {}
    torch_mod.save          = lambda *a, **kw: None
    torch_mod.no_grad       = lambda: type("_NG", (), {
        "__enter__": lambda s: None,
        "__exit__":  lambda s, *a: None,
    })()
    torch_mod.zeros         = lambda *a, **kw: None
    torch_mod.tensor        = lambda *a, **kw: None
    torch_mod.long          = 0
    torch_mod.float         = 1
    torch_mod.manual_seed   = lambda s: None
    torch_mod.Tensor        = type("Tensor", (), {})  # type stub for annotations

    sys.modules["torch"]               = torch_mod
    sys.modules["torch.nn"]            = nn_mod
    sys.modules["torch.nn.functional"] = nn_functional
    # Extra submodules that noisereduce/torchgate imports
    types_mod = types.ModuleType("torch.types")
    types_mod.Number = (int, float)   # noisereduce.torchgate.utils imports this
    types_mod._int   = int
    types_mod._float = float
    types_mod._bool  = bool
    sys.modules["torch.types"] = types_mod


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use the default asyncio event loop policy."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()
