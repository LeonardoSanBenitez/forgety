"""
Pytest configuration for forgety tests.

Sets rootdir to /tests and ensures /src is on sys.path so that
'app.main' and 'libs.*' are importable from the test environment.
"""
import os
import sys

# Ensure /src is on the path so 'from app.main import ...' works
if '/src' not in sys.path:
    sys.path.insert(0, '/src')


def pytest_configure(config: "object") -> None:
    """Register custom markers.

    ``gpu`` marks tests that require a CUDA/ROCm device. They are excluded
    from the default ``make test`` run (GitHub Actions running the
    vision-unlearning tests has no GPU) and run explicitly via
    ``make test-gpu`` / ``pytest -m gpu``.
    """
    config.addinivalue_line(  # type: ignore[attr-defined]
        "markers",
        "gpu: test requires a GPU (CUDA/ROCm); excluded from default run",
    )
