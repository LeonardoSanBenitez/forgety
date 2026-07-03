"""
Offline tests for the backend's I-CARE routes:

- POST /v1/public-api-compute-rt
- GET  /v1/public-api-read-interference-per-entity-all
- GET  /v1/public-api-entity-model-metrics

All tests run without network access. HuggingFace helpers are monkeypatched to fail
loudly if anything attempts a download, and the routes are fed from synthetic files
written under a temporary working directory (the I-CARE loaders read relative
'assets/...' paths, so a chdir into tmp_path exercises the real local-file code path).

The fixture values (see icare_fixtures.py, shared with the UI tests) are adversarial:
every value is derived from the identity of the entity it belongs to, and the expected
values asserted below are recomputed independently from the same identity function.
"""
from typing import Any, Dict, Generator, List

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_database, get_infra
from libs.database import DatabaseLocalJson
from libs.infra.fake import InfraFake

from icare_fixtures import (
    TASKS,
    ENTITIES,
    entity_value,
    pair_value,
    write_interference_per_entity_files,
    write_interference_matrix_result,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _fail_on_network(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("Network access attempted in an offline test (HuggingFace helper was called).")


@pytest.fixture
def icare_assets(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    """Synthetic I-CARE asset tree in a temporary CWD, with all HF access blocked."""
    monkeypatch.chdir(tmp_path)
    assets_dir = str(tmp_path / "assets")
    write_interference_per_entity_files(assets_dir)
    matrix_data = write_interference_matrix_result(assets_dir)

    import vision_unlearning.benchmarks.I_care.metadata as icare_metadata
    import vision_unlearning.benchmarks.I_care.result_templates as icare_result_templates
    for module in (icare_metadata, icare_result_templates):
        monkeypatch.setattr(module, "huggingface_dataset_file_exists", _fail_on_network)
        monkeypatch.setattr(module, "huggingface_dataset_file_download", _fail_on_network)

    return {"matrix_data": matrix_data}


@pytest.fixture
def client(icare_assets: Dict[str, Any]) -> Generator[TestClient, None, None]:
    """TestClient with InfraFake + ephemeral DatabaseLocalJson (same doubles as test_api.py)."""
    app.dependency_overrides[get_database] = lambda: DatabaseLocalJson(filepath="test_db.json")
    app.dependency_overrides[get_infra] = lambda: InfraFake()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /v1/public-api-read-interference-per-entity-all
# ---------------------------------------------------------------------------

def test_read_interference_all_tasks_present(client: TestClient) -> None:
    """The route returns one entry per I-CARE task, keyed by the backend task name."""
    response = client.get("/v1/public-api-read-interference-per-entity-all")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == set(TASKS)


def test_read_interference_entity_values_are_correct(client: TestClient) -> None:
    """Every entity carries exactly its own identity-derived value.

    A scrambled entity mapping or a task mix-up returns a wrong number here, because
    the expected value is recomputed independently per (task, entity)."""
    response = client.get("/v1/public-api-read-interference-per-entity-all")
    assert response.status_code == 200
    data = response.json()
    for task in TASKS:
        records = data[task]
        assert [r["name"] for r in records] == ENTITIES[task]
        for i, record in enumerate(records):
            assert record["metric_emitter_average_clip_diff_20"] == entity_value(task, i)


def test_read_interference_tasks_are_isolated(client: TestClient) -> None:
    """No value from one task appears in another task's records."""
    response = client.get("/v1/public-api-read-interference-per-entity-all")
    data = response.json()
    values_per_task = {
        task: {r["metric_emitter_average_clip_diff_20"] for r in data[task]}
        for task in TASKS
    }
    for task_a in TASKS:
        for task_b in TASKS:
            if task_a != task_b:
                assert values_per_task[task_a].isdisjoint(values_per_task[task_b])


# ---------------------------------------------------------------------------
# POST /v1/public-api-compute-rt
# ---------------------------------------------------------------------------

def test_compute_rt_interference_matrix_gui_params(client: TestClient, icare_assets: Dict[str, Any]) -> None:
    """GUI-named parameters resolve to the correct backend RT result.

    The request uses the exact display names the frontend sends ('Stable Diffusion 1.4',
    'People', 'UCE', 'Delta Clip'). The pre-computed result file only exists at the path
    produced by the correct backend conversion (sd1.4/people/uce/clip_diff), so any
    parameter scrambling makes this test fail."""
    response = client.post(
        "/v1/public-api-compute-rt",
        json={
            "template": "InterferenceMatrix",
            "params": {
                "model": "Stable Diffusion 1.4",
                "task": "People",
                "unlearning_algorithm": "UCE",
                "interference_pair": "Delta Clip",
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data == icare_assets["matrix_data"]


def test_compute_rt_pair_values_and_direction(client: TestClient) -> None:
    """Exact per-pair values and the known clip_diff metric direction.

    clip_diff's direction is '↑' (more interference = more negative; see
    configuration.mp_to_direction). A flipped direction or a transposed emitter/receiver
    matrix produces a wrong value here."""
    response = client.post(
        "/v1/public-api-compute-rt",
        json={
            "template": "InterferenceMatrix",
            "params": {
                "model": "Stable Diffusion 1.4",
                "task": "People",
                "unlearning_algorithm": "UCE",
                "interference_pair": "Delta Clip",
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["metric_direction"] == "↑"
    entities = ENTITIES["people"]
    rows = {row["emitter"]: row for row in data["result"]}
    assert set(rows.keys()) == set(entities)
    # Spot-check an asymmetric off-diagonal pair: (emitter=1, receiver=2) != (emitter=2, receiver=1).
    assert rows["Alan Turing"]["Grace Hopper"] == pair_value(1, 2)
    assert rows["Grace Hopper"]["Alan Turing"] == pair_value(2, 1)
    assert pair_value(1, 2) != pair_value(2, 1)


def test_compute_rt_unknown_template_is_an_error(icare_assets: Dict[str, Any]) -> None:
    """An unknown template name is a server error (current contract), never a silent 200."""
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/public-api-compute-rt",
            json={"template": "NoSuchTemplate", "params": {}},
        )
    assert response.status_code == 500


# ---------------------------------------------------------------------------
# GET /v1/public-api-entity-model-metrics
# ---------------------------------------------------------------------------

def test_entity_model_metrics_model_id_construction(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The HF model id is built from (method, task, entity slug) exactly, and the
    metrics returned by the lookup are passed through unmodified."""
    seen_model_ids: List[str] = []

    def fake_metrics(model_id: str) -> Dict[str, Any]:
        seen_model_ids.append(model_id)
        return {"model_id": model_id, "fid": 42.5}

    import app.main as backend_main
    monkeypatch.setattr(backend_main, "huggingface_get_model_metrics", fake_metrics)

    response = client.get(
        "/v1/public-api-entity-model-metrics",
        params={"task": "people", "entity": "Ada Lovelace", "unlearning_algorithm": "distil"},
    )
    assert response.status_code == 200
    assert seen_model_ids == ["LeonardoBenitez/VisionUnlearning-distil-people-ada-lovelace"]
    assert response.json() == {"model_id": "LeonardoBenitez/VisionUnlearning-distil-people-ada-lovelace", "fid": 42.5}


def test_entity_model_metrics_lookup_failure_returns_empty_dict(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing HF lookup degrades to an empty dict (the frontend's warning path), not a 500."""
    def failing_metrics(model_id: str) -> Dict[str, Any]:
        raise RuntimeError("model not found")

    import app.main as backend_main
    monkeypatch.setattr(backend_main, "huggingface_get_model_metrics", failing_metrics)

    response = client.get(
        "/v1/public-api-entity-model-metrics",
        params={"task": "people", "entity": "Ada Lovelace", "unlearning_algorithm": "uce"},
    )
    assert response.status_code == 200
    assert response.json() == {}
