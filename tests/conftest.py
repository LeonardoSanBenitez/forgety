"""
Pytest configuration for forgety tests.

Sets rootdir to /tests and ensures 'app.main' and 'libs.*' are importable from the test
environment, whether running inside the Docker container (where the backend's
docker-compose.yml bind-mounts services/backend/app/ and libs/ under /src) or on the host
/ in CI (no such mount exists, so both source roots are added explicitly instead).
"""
import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)  # forgety/
# 'app' lives under services/backend/, not at the repo root -- its PARENT directory must
# be on sys.path for 'from app.main import ...' to resolve.
_BACKEND_APP_PARENT = os.path.join(_REPO_ROOT, "services", "backend")

if os.path.isdir('/src'):
    # Docker container: docker-compose.yml bind-mounts app/ and libs/ directly under /src.
    if '/src' not in sys.path:
        sys.path.insert(0, '/src')
else:
    # Host / CI: no /src mount. 'app' and 'libs' are not siblings in the repo layout, so
    # both roots are added explicitly.
    for _path in (_REPO_ROOT, _BACKEND_APP_PARENT):
        if _path not in sys.path:
            sys.path.insert(0, _path)


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
