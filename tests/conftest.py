"""Shared test setup. CUDA test cases are opt-in: set SC_TEST_CUDA=1 (they allocate GPU memory and can stall a
running training). Without it they are skipped; CPU torch cases always run."""
import os

import pytest

CUDA_REASON = "cuda tests are opt-in: set SC_TEST_CUDA=1 (needs a free GPU, not during a training run)"
CUDA_ENABLED = os.environ.get("SC_TEST_CUDA", "") == "1"

# decorator for cuda-only tests, parametrize list for tests that run on cpu and (opt-in) cuda
CUDA = pytest.mark.cuda
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: needs a GPU; runs only with SC_TEST_CUDA=1")


def pytest_collection_modifyitems(config, items):
    skip_off = pytest.mark.skip(reason=CUDA_REASON)
    skip_none = pytest.mark.skip(reason="cuda not available")
    avail = None
    for item in items:
        if "cuda" not in item.keywords:
            continue
        if not CUDA_ENABLED:
            item.add_marker(skip_off)
            continue
        if avail is None:
            import torch
            avail = torch.cuda.is_available()
        if not avail:
            item.add_marker(skip_none)
