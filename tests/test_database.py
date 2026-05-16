import json
import os
import tempfile
import threading
import pytest

from libs.database import DatabaseLocalJson
from libs.schemas import RequestLaunched


def _make_launched(**kwargs: object) -> RequestLaunched:
    defaults = dict(
        customer_id="cust-1",
        experiment_name="exp",
        model_base_name="SD",
        concept_forget="parachute",
        concept_overwrite="airplane",
        concept_retain=None,
        unlearning_algorithm="FADE",
        model_output_hf_id="somewhere",
        hyperparameters={"epochs": 5},
        num_forget_images=10,
        num_retain_images=10,
        timestamp_started=1234567890,
        slurm_job_id="job-1",
    )
    defaults.update(kwargs)  # type: ignore[arg-type]
    return RequestLaunched(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def db(tmp_path: object) -> DatabaseLocalJson:
    import pathlib
    fp = str(pathlib.Path(str(tmp_path)) / "db.json")
    return DatabaseLocalJson(filepath=fp)


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------

def test_empty_on_start(db: DatabaseLocalJson) -> None:
    assert db.get_requests() == []


def test_insert_and_get(db: DatabaseLocalJson) -> None:
    req = _make_launched()
    db.insert_request(req)

    result = db.get_request(uuid=req.uuid)
    assert result is not None
    assert result.uuid == req.uuid


def test_insert_and_list(db: DatabaseLocalJson) -> None:
    req = _make_launched()
    db.insert_request(req)

    results = db.get_requests()
    assert len(results) == 1


def test_get_requests_status_none(db: DatabaseLocalJson) -> None:
    req = _make_launched()
    db.insert_request(req)

    pending = db.get_requests_status_none()
    assert len(pending) == 1


def test_update_sets_status(db: DatabaseLocalJson) -> None:
    req = _make_launched()
    db.insert_request(req)

    db.update_request(uuid=req.uuid, updated_request={"status": "SUCCEEDED"})

    result = db.get_request(uuid=req.uuid)
    assert result is not None
    assert result.status == "SUCCEEDED"  # type: ignore[union-attr]


def test_update_clears_status_none_list(db: DatabaseLocalJson) -> None:
    req = _make_launched()
    db.insert_request(req)
    db.update_request(uuid=req.uuid, updated_request={"status": "SUCCEEDED"})

    pending = db.get_requests_status_none()
    assert len(pending) == 0


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_update_nonexistent_uuid_raises(db: DatabaseLocalJson) -> None:
    req = _make_launched()
    db.insert_request(req)

    with pytest.raises(ValueError, match="not found"):
        db.update_request(uuid="does-not-exist", updated_request={"status": "SUCCEEDED"})


def test_get_nonexistent_uuid_returns_none(db: DatabaseLocalJson) -> None:
    req = _make_launched()
    db.insert_request(req)

    result = db.get_request(uuid="does-not-exist")
    assert result is None


def test_get_request_missing_file_returns_none(db: DatabaseLocalJson) -> None:
    # File never created — get_request should return None, not raise
    result = db.get_request(uuid="any-uuid")
    assert result is None


def test_corrupted_json_raises(db: DatabaseLocalJson) -> None:
    # Write invalid JSON, then try to read
    req = _make_launched()
    db.insert_request(req)  # creates the file

    with open(db.filepath, "w") as f:
        f.write("this is not valid json {{{{")

    with pytest.raises(Exception):
        db.get_requests()


# ---------------------------------------------------------------------------
# Concurrent access
# ---------------------------------------------------------------------------

def test_concurrent_inserts(db: DatabaseLocalJson) -> None:
    """Two threads writing simultaneously must not corrupt the file."""
    errors: list = []

    def insert_one(i: int) -> None:
        try:
            req = _make_launched(
                customer_id=f"cust-{i}",
                experiment_name=f"exp-{i}",
                slurm_job_id=f"job-{i}",
            )
            db.insert_request(req)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=insert_one, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Concurrent insert raised errors: {errors}"
    results = db.get_requests()
    assert len(results) == 10
