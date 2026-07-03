"""
Pytest configuration for forgety tests.

Sets rootdir to /tests and ensures 'app.main', 'libs.*' and 'vision_unlearning.*' are
importable from the test environment, whether running inside the Docker container (where
the backend's docker-compose.yml bind-mounts services/backend/app/ and libs/ under /src
and ../vision-unlearning under /vision_unlearning_src, both on PYTHONPATH) or on the host
/ in CI (no such mounts exist, so the source roots are added explicitly instead).
"""
import importlib.util
import os
import sys

# Force matplotlib's non-interactive Agg backend before it is ever imported: the UI tests
# render Result Template figures inside AppTest's script thread, and interactive backends
# (TkAgg, the default on desktop hosts) crash the process when their objects are created
# or garbage-collected outside the main thread.
os.environ.setdefault("MPLBACKEND", "Agg")

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)  # forgety/
# 'app' lives under services/backend/, not at the repo root -- its PARENT directory must
# be on sys.path for 'from app.main import ...' to resolve.
_BACKEND_APP_PARENT = os.path.join(_REPO_ROOT, "services", "backend")
# vision-unlearning is not yet installed as a package (see docker-compose.yml). On the
# host and in CI the repo is expected as a sibling checkout of forgety/, mirroring the
# container's /vision_unlearning_src bind-mount.
_VISION_UNLEARNING_SIBLING = os.path.join(os.path.dirname(_REPO_ROOT), "vision-unlearning")

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

if importlib.util.find_spec("vision_unlearning") is None and os.path.isdir(_VISION_UNLEARNING_SIBLING):
    sys.path.insert(0, _VISION_UNLEARNING_SIBLING)


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
    config.addinivalue_line(  # type: ignore[attr-defined]
        "markers",
        "integration: test downloads real I-CARE data from HuggingFace; excluded from the "
        "default offline run and executed by the scheduled integration workflow",
    )
