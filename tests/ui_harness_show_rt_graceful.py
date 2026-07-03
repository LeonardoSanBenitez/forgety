"""Minimal Streamlit harness that renders the frontend and then calls
show_rt_graceful directly.

Why this exists: show_rt_graceful's yellow-warning fallback is normally reachable only
inside the entity-detail dialog on the 'Explore results' page, which opens on a
st.dataframe row selection -- an interaction streamlit.testing.v1.AppTest cannot drive.
Running this harness under AppTest exercises the function's three outcomes (plot on 200,
warning on non-200, warning on connection failure) without the dialog.
"""
import os
import sys

_FRONTEND_APP_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "services", "frontend", "app")
)
if _FRONTEND_APP_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_APP_DIR)

import main as frontend_main  # noqa: E402  (imports must follow the sys.path setup)

frontend_main.show_rt_graceful(
    template="InterferenceMatrix",
    model="Stable Diffusion 1.4",
    task="People",
    params={"unlearning_algorithm": "UCE", "interference_pair": "Delta Clip"},
)
