"""
Integration tests against the REAL I-CARE data hosted on HuggingFace
(LeonardoBenitez/VisionUnlearningEvaluationTestbeds).

These tests exist to catch drift in the HF-hosted data/schema that the offline,
fixture-based tests structurally cannot see. They download real files, so they are:

- marked `integration` and excluded from every default run (`make test`, the offline CI
  job) -- see conftest.py;
- executed by the scheduled integration workflow (weekly cron + manual trigger), where a
  failure means "the published data no longer matches what the application expects",
  not "a commit broke the code".

One task slice (people) keeps the download volume small. Assertions target the shape
and the stable, load-bearing facts the frontend relies on (task keys, 100 entities,
name/metric columns, matrix dimensions, metric direction) rather than exact float
values, which legitimately change when the data is regenerated.
"""
from typing import Any, Dict, Generator

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_database, get_infra
from libs.database import DatabaseLocalJson
from libs.infra.fake import InfraFake

pytestmark = pytest.mark.integration

TASKS = ["breeds", "scenes", "people"]
N_ENTITIES_PEOPLE = 100
KNOWN_PEOPLE_ENTITY = "George_W_Bush"


@pytest.fixture
def client(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    """TestClient with the usual test doubles, chdir'd to a tmp CWD so all HF downloads
    land in an ephemeral assets/ tree instead of polluting the repository."""
    monkeypatch.chdir(tmp_path)
    app.dependency_overrides[get_database] = lambda: DatabaseLocalJson(filepath="test_db.json")
    app.dependency_overrides[get_infra] = lambda: InfraFake()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_list_entities_real_data(client: TestClient) -> None:
    """The entity-listing route works end to end against the published data: all three
    tasks present, 100 unique people entities including a known one, and the metric
    columns the frontend's entity detail panel reads."""
    response = client.get("/v1/public-api-read-interference-per-entity-all")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == set(TASKS)

    people = data["people"]
    assert len(people) == N_ENTITIES_PEOPLE
    names = [record["name"] for record in people]
    assert len(set(names)) == len(names)
    assert KNOWN_PEOPLE_ENTITY in names
    metric_columns = [k for k in people[0].keys() if k.startswith("metric_")]
    assert len(metric_columns) > 0


def test_compute_rt_interference_matrix_real_data(client: TestClient) -> None:
    """InterferenceMatrix end to end with the frontend's GUI parameter names: the
    published result has the full 100x100 entity grid and the correct clip_diff
    metric direction."""
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
    assert data["metadata"]["task"] == "people"
    assert data["metadata"]["unlearning_algorithm"] == "uce"
    assert data["metadata"]["interference_pair"] == "clip_diff"
    assert data["metadata"]["metric_direction"] == "↑"

    rows = data["result"]
    assert len(rows) == N_ENTITIES_PEOPLE
    emitters = [row["emitter"] for row in rows]
    assert KNOWN_PEOPLE_ENTITY in emitters
    for row in rows:
        # emitter + one column per receiver entity
        assert len(row) == N_ENTITIES_PEOPLE + 1


def test_compute_rt_similarity_matrix_real_data(client: TestClient) -> None:
    """SimilarityMatrix end to end: full grid, published clip similarities are cosines
    scaled to [-100, 100], and every entity's self-similarity is the known exact value
    (~100, cosine of an embedding with itself) and bounds its whole row."""
    response = client.post(
        "/v1/public-api-compute-rt",
        json={
            "template": "SimilarityMatrix",
            "params": {
                "model": "Stable Diffusion 1.4",
                "task": "People",
                "similarity_metric": "Clip Cosine Similarity",
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    rows = data["result"]
    assert len(rows) == N_ENTITIES_PEOPLE
    label_key = next(k for k in rows[0].keys() if not isinstance(rows[0][k], (int, float)))
    for row in rows:
        own_name = row[label_key]
        self_similarity = row[own_name]
        assert abs(self_similarity - 100.0) < 0.1, f"self-similarity of {own_name} is not ~100: {self_similarity}"
        for key, value in row.items():
            if isinstance(value, (int, float)):
                assert -100.1 <= value <= self_similarity + 0.1, f"similarity out of range: {key}={value}"


def test_entity_model_metrics_real_lookup_is_graceful(client: TestClient) -> None:
    """The real HF model lookup never breaks the route: it returns a dict (currently
    empty, because the per-entity unlearned models are not published on HuggingFace)."""
    response = client.get(
        "/v1/public-api-entity-model-metrics",
        params={"task": "people", "entity": "George W Bush", "unlearning_algorithm": "uce"},
    )
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
