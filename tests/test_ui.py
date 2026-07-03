"""
UI tests for the Streamlit frontend (services/frontend/app/main.py), using Streamlit's
native in-process test harness (streamlit.testing.v1.AppTest) -- no browser, no running
backend, no network.

All backend HTTP calls (http://backend:80/...) are mocked with the `responses` library.
The mocked payloads are the SAME adversarial fixtures used by the backend route tests
(icare_fixtures.py): every value is derived from the identity of the entity it belongs
to, so a scrambled entity mapping or a flipped sign renders a visibly wrong number that
the exact-value assertions below catch -- these are regression tests, not render checks.

Known coverage limit: the entity-detail dialog on 'Explore results' opens on a
st.dataframe row selection, which AppTest cannot drive. The dialog's data logic is
covered by the backend route tests, and its show_rt_graceful fallback is covered here
through tests/ui_harness_show_rt_graceful.py.

These tests require streamlit and responses; they skip cleanly in environments without
them (e.g. the backend Docker container used by `make test`).
"""
import os
from typing import Any, Dict, List
from urllib.parse import parse_qs

import pytest

pytest.importorskip("streamlit", reason="UI tests need streamlit (absent in the backend container)")
pytest.importorskip("responses", reason="UI tests need responses for HTTP mocking")

import responses  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from icare_fixtures import (  # noqa: E402
    ENTITIES,
    build_interference_matrix_data,
    build_interference_per_entity_data,
    entity_value,
)

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_MAIN = os.path.abspath(os.path.join(_TESTS_DIR, "..", "services", "frontend", "app", "main.py"))
HARNESS_SHOW_RT_GRACEFUL = os.path.join(_TESTS_DIR, "ui_harness_show_rt_graceful.py")

PAGES = ["Home", "Create a request", "List requests", "Explore results", "Compute RT"]

BACKEND = "http://backend:80"


def _app(page: str | None = None) -> AppTest:
    at = AppTest.from_file(FRONTEND_MAIN, default_timeout=60)
    if page is not None:
        at.session_state["page"] = page
    return at


# ---------------------------------------------------------------------------
# Home + navigation
# ---------------------------------------------------------------------------

def test_home_renders_without_exception_and_shows_navigation() -> None:
    at = _app().run()
    assert not at.exception
    assert [b.label for b in at.button] == PAGES
    assert any("Machine Unlearning" in str(m.value) for m in at.markdown)


def test_navigation_button_switches_page() -> None:
    at = _app().run()
    at.button[PAGES.index("Create a request")].click().run()
    assert not at.exception
    assert at.session_state["page"] == "Create a request"
    assert at.title[0].value == "Create a Request"


# ---------------------------------------------------------------------------
# Create a request
# ---------------------------------------------------------------------------

def test_create_request_missing_required_fields_shows_error() -> None:
    at = _app("Create a request").run()
    submit = [b for b in at.button if b.label == "Send"]
    assert len(submit) == 1
    submit[0].click().run()
    assert not at.exception
    assert len(at.error) == 1
    for field in ["Experiment Name", "Model Base Name", "Model Output HF ID"]:
        assert field in str(at.error[0].value)


@responses.activate(assert_all_requests_are_fired=False)
def test_create_request_sends_exact_form_values_and_switches_page() -> None:
    responses.add(responses.POST, f"{BACKEND}/v1/public-api-create-request", json="11111111-2222-3333-4444-555555555555")
    responses.add(responses.GET, f"{BACKEND}/v1/public-api-read-requests", json=[])

    at = _app("Create a request").run()
    # Field order in the form: experiment_name, model_base_name, concept_forget,
    # concept_overwrite, concept_retain, model_output_hf_id.
    at.text_input[0].input("experiment-alpha")
    at.text_input[1].input("model-base-alpha")
    at.text_input[2].input("concept-forget-alpha")
    at.text_input[5].input("hf-id-alpha")
    at.selectbox[0].select("UCE")
    [b for b in at.button if b.label == "Send"][0].click().run()

    assert not at.exception
    assert at.session_state["page"] == "List requests"

    create_calls = [c for c in responses.calls if c.request.url is not None and c.request.url.endswith("/v1/public-api-create-request")]
    assert len(create_calls) == 1
    sent = parse_qs((create_calls[0].request.body or "") if isinstance(create_calls[0].request.body, str) else (create_calls[0].request.body or b"").decode())
    assert sent["experiment_name"] == ["experiment-alpha"]
    assert sent["model_base_name"] == ["model-base-alpha"]
    assert sent["concept_forget"] == ["concept-forget-alpha"]
    assert sent["model_output_hf_id"] == ["hf-id-alpha"]
    assert sent["unlearning_algorithm"] == ["UCE"]


# ---------------------------------------------------------------------------
# List requests
# ---------------------------------------------------------------------------

def _two_requests_payload() -> List[Dict[str, Any]]:
    """Two requests whose field values encode their own identity, plus one metric pair
    with a known exact rendering: 'clip mean score' 3.96 ± 'clip std score' 0.13 must
    render as '3.96 ± 0.1' (a negation would render '-3.96', a reciprocal '0.25')."""
    return [
        {
            "experiment_name": "experiment-alpha",
            "model_base_name": "model-alpha",
            "concept_forget": "forget-alpha",
            "unlearning_algorithm": "UCE",
            "model_output_hf_id": "hf-alpha",
            "status": "completed",
            "metrics": [
                {"name": "clip mean score", "value": 3.96},
                {"name": "clip std score", "value": 0.13},
            ],
        },
        {
            "experiment_name": "experiment-beta",
            "model_base_name": "model-beta",
            "concept_forget": "forget-beta",
            "unlearning_algorithm": "FADE",
            "model_output_hf_id": "hf-beta",
            "status": "running",
        },
    ]


@responses.activate(assert_all_requests_are_fired=False)
def test_list_requests_renders_each_request_with_its_own_values() -> None:
    responses.add(responses.GET, f"{BACKEND}/v1/public-api-read-requests", json=_two_requests_payload())
    at = _app("List requests").run()
    assert not at.exception

    blocks = [str(m.value) for m in at.markdown]
    alpha_blocks = [b for b in blocks if "experiment-alpha" in b]
    beta_blocks = [b for b in blocks if "experiment-beta" in b]
    assert len(alpha_blocks) == 1
    assert len(beta_blocks) == 1
    # Each request's block carries its OWN concept and algorithm, not its neighbour's.
    assert "forget-alpha" in alpha_blocks[0] and "UCE" in alpha_blocks[0]
    assert "forget-beta" in beta_blocks[0] and "FADE" in beta_blocks[0]
    assert "forget-beta" not in alpha_blocks[0]
    assert "forget-alpha" not in beta_blocks[0]
    # Known exact value: mean/std pair folded into one line, correct sign and magnitude.
    assert "**clip score**: 3.96 ± 0.1" in alpha_blocks[0]


# ---------------------------------------------------------------------------
# Explore results
# ---------------------------------------------------------------------------

@responses.activate(assert_all_requests_are_fired=False)
def test_explore_results_shows_entities_of_the_selected_task() -> None:
    responses.add(responses.GET, f"{BACKEND}/v1/public-api-read-interference-per-entity-all", json=build_interference_per_entity_data())
    at = _app("Explore results").run()
    assert not at.exception
    assert at.selectbox[1].value == "Breeds"

    df = at.dataframe[0].value
    assert list(df["name"]) == ENTITIES["breeds"]
    assert list(df["metric_emitter_average_clip_diff_20"]) == [entity_value("breeds", i) for i in range(len(ENTITIES["breeds"]))]


@responses.activate(assert_all_requests_are_fired=False)
def test_explore_results_task_switch_shows_that_tasks_data() -> None:
    """Selecting 'People' must show people entities with people values -- catches a
    GUI-name-to-backend-key mapping regression between tasks."""
    responses.add(responses.GET, f"{BACKEND}/v1/public-api-read-interference-per-entity-all", json=build_interference_per_entity_data())
    at = _app("Explore results").run()
    at.selectbox[1].select("People").run()
    assert not at.exception

    df = at.dataframe[0].value
    assert list(df["name"]) == ENTITIES["people"]
    assert list(df["metric_emitter_average_clip_diff_20"]) == [entity_value("people", i) for i in range(len(ENTITIES["people"]))]


@responses.activate(assert_all_requests_are_fired=False)
def test_explore_results_backend_error_shows_error() -> None:
    responses.add(responses.GET, f"{BACKEND}/v1/public-api-read-interference-per-entity-all", status=503)
    at = _app("Explore results").run()
    assert not at.exception
    assert len(at.error) == 1
    assert "503" in str(at.error[0].value)


# ---------------------------------------------------------------------------
# Compute RT
# ---------------------------------------------------------------------------

@responses.activate(assert_all_requests_are_fired=False)
def test_compute_rt_sends_exact_request_and_plots_result() -> None:
    """Submitting the form sends exactly the selected GUI parameters, and a 200 response
    is rendered through the real RT plot code (not a stub)."""
    responses.add(responses.POST, f"{BACKEND}/v1/public-api-compute-rt", json=build_interference_matrix_data())

    at = _app("Compute RT").run()
    at.selectbox[0].select("InterferenceMatrix").run()
    at.selectbox[2].select("People")
    at.selectbox[3].select("UCE")
    at.selectbox[4].select("Delta Clip")
    [b for b in at.button if b.label == "Compute"][0].click().run()

    assert not at.exception
    assert len(at.success) == 1

    compute_calls = [c for c in responses.calls if c.request.url is not None and c.request.url.endswith("/v1/public-api-compute-rt")]
    assert len(compute_calls) == 1
    import json as _json
    body = compute_calls[0].request.body
    sent = _json.loads(body if isinstance(body, str) else (body or b"").decode())
    assert sent == {
        "template": "InterferenceMatrix",
        "params": {
            "model": "Stable Diffusion 1.4",
            "task": "People",
            "unlearning_algorithm": "UCE",
            "interference_pair": "Delta Clip",
        },
    }


@responses.activate(assert_all_requests_are_fired=False)
def test_compute_rt_backend_error_shows_error() -> None:
    responses.add(responses.POST, f"{BACKEND}/v1/public-api-compute-rt", status=500, body="boom")
    at = _app("Compute RT").run()
    at.selectbox[0].select("InterferenceMatrix").run()
    [b for b in at.button if b.label == "Compute"][0].click().run()
    assert not at.exception
    assert len(at.error) == 1
    assert "Backend error: 500" in str(at.error[0].value)


# ---------------------------------------------------------------------------
# show_rt_graceful (via the harness; see module docstring for why)
# ---------------------------------------------------------------------------

@responses.activate(assert_all_requests_are_fired=False)
def test_show_rt_graceful_plots_on_success() -> None:
    responses.add(responses.POST, f"{BACKEND}/v1/public-api-compute-rt", json=build_interference_matrix_data())
    at = AppTest.from_file(HARNESS_SHOW_RT_GRACEFUL, default_timeout=60).run()
    assert not at.exception
    assert len(at.warning) == 0


@responses.activate(assert_all_requests_are_fired=False)
def test_show_rt_graceful_warns_on_backend_error() -> None:
    responses.add(responses.POST, f"{BACKEND}/v1/public-api-compute-rt", status=503)
    at = AppTest.from_file(HARNESS_SHOW_RT_GRACEFUL, default_timeout=60).run()
    assert not at.exception
    assert len(at.warning) == 1
    assert "503" in str(at.warning[0].value)


@responses.activate(assert_all_requests_are_fired=False)
def test_show_rt_graceful_warns_on_connection_failure() -> None:
    # No mock registered for the compute-rt URL: responses raises ConnectionError,
    # which the function must degrade to a warning, never an exception.
    at = AppTest.from_file(HARNESS_SHOW_RT_GRACEFUL, default_timeout=60).run()
    assert not at.exception
    assert len(at.warning) == 1
    assert "unavailable" in str(at.warning[0].value)
