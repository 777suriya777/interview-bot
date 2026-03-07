"""
Minimal torch stub for testing FastAPI endpoints without the real torch package.
Import this module at the top of any test file that needs to import torch-dependent
service code under test.
"""
import sys
import types


def install_torch_stub():
    """Register a minimal torch stub in sys.modules if torch is not installed."""
    if "torch" in sys.modules:
        return  # real torch already present

    # ── Build stub modules ───────────────────────────────────────────
    torch_mod = types.ModuleType("torch")

    # nn.Module base class — needs to be a real Python class so subclassing works
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

    torch_mod.nn            = nn_mod
    torch_mod.device        = lambda s: s
    torch_mod.cuda          = types.SimpleNamespace(is_available=lambda: False)
    torch_mod.load          = lambda *a, **kw: {}
    torch_mod.save          = lambda *a, **kw: None
    torch_mod.no_grad       = lambda: type("_NG", (), {
        "__enter__": lambda s: None,
        "__exit__": lambda s, *a: None,
    })()
    torch_mod.zeros         = lambda *a, **kw: None
    torch_mod.tensor        = lambda *a, **kw: None
    torch_mod.long          = 0
    torch_mod.float         = 1
    torch_mod.manual_seed   = lambda s: None

    sys.modules["torch"]             = torch_mod
    sys.modules["torch.nn"]          = nn_mod
    sys.modules["torch.nn.functional"] = nn_functional
