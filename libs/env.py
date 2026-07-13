"""Environment-variable normalization shared by forgety's three Docker entrypoints.

Import this module FIRST in every service entrypoint (``services/backend/app/main.py``,
``services/frontend/app/main.py``, ``services/mcp/app/server.py``), before any other
import that might read ``HF_TOKEN`` -- including the ``vision_unlearning`` imports that
follow it. A plain ``import libs.env`` is enough: the normalization below runs as an
import-time side effect, no explicit function call needed at each call site.

Why this exists:
Docker's ``env_file`` sets a container env var verbatim from ``.env``. A colleague who
copies ``.env.template`` as-is (the only documented action today) gets ``HF_TOKEN=``,
which becomes an *empty string* in the container environment, not "unset". Downstream,
``huggingface_hub`` builds the request header as ``f"Bearer {token}"``; an empty string
produces the illegal header ``"Bearer "`` and breaks *anonymous* access to public HF
repos -- even though anonymous access (``token=None``) works fine and is exactly what
the entity-listing / run-RT features are meant to use with no ``.env`` at all.

``os.getenv('HF_TOKEN')`` must return ``None``, not ``""``, for the existing
local -> HF -> raise fallback in vision_unlearning
(``metadata.py:171``, ``result_templates.py:176``) to behave as its own comments say it
should. The root cause lives in vision_unlearning, but per project decision this is
fixed at the forgety boundary instead of editing vision_unlearning's source (which is
mounted read-only from a sibling checkout, not owned by this repo).
"""
import os

if not os.environ.get("HF_TOKEN"):
    os.environ.pop("HF_TOKEN", None)
