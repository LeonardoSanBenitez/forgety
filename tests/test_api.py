"""
API tests for the FastAPI backend.

All tests use TestClient with dependency injection overrides to replace the
real InfraSlurm and DatabaseLocalJson with test doubles.  No Slurm cluster,
no persistent database file.

The test_create_request_fade_happy_path test writes a zip to /requests/ inside
the container.  This directory is created by the fixture.
"""
import io
import json
import os
import struct
import tempfile
import zipfile
import zlib
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_database, get_infra
from libs.database import DatabaseLocalJson
from libs.infra.fake import InfraFake


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png(color: tuple = (255, 0, 0)) -> bytes:
    """Return bytes of a minimal 1x1 PNG (no PIL dependency)."""
    width, height = 1, 1

    def chunk(tag: bytes, data: bytes) -> bytes:
        payload = tag + data
        return struct.pack('>I', len(data)) + payload + struct.pack('>I', zlib.crc32(payload) & 0xffffffff)

    signature = b'\x89PNG\r\n\x1a\n'
    ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    raw = b'\x00' + bytes(color)
    idat = chunk(b'IDAT', zlib.compress(raw))
    iend = chunk(b'IEND', b'')
    return signature + ihdr + idat + iend


def _make_dataset_zip(include_forget: bool = True, include_retain: bool = True) -> bytes:
    """Return bytes of a zip containing forget/ and/or retain/ folders."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        if include_forget:
            zf.writestr('forget/img.png', _make_png((255, 0, 0)))
        if include_retain:
            zf.writestr('retain/img.png', _make_png((0, 255, 0)))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path: object) -> DatabaseLocalJson:
    import pathlib
    fp = str(pathlib.Path(str(tmp_path)) / "test_db.json")
    return DatabaseLocalJson(filepath=fp)


@pytest.fixture
def client(tmp_db: DatabaseLocalJson) -> Generator[TestClient, None, None]:
    """TestClient with InfraFake + ephemeral DatabaseLocalJson."""
    fake_infra = InfraFake()
    app.dependency_overrides[get_database] = lambda: tmp_db
    app.dependency_overrides[get_infra] = lambda: fake_infra
    # Ensure /requests/ exists inside the container for FADE zip extraction
    os.makedirs('/requests', exist_ok=True)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_availability(client: TestClient) -> None:
    """POST /v1/availability-test returns 200."""
    response = client.post("/v1/availability-test")
    assert response.status_code == 200
    assert response.json() == 200


def test_create_request_uce(client: TestClient) -> None:
    """UCE path: no dataset upload, concept_forget + concept_retain required."""
    response = client.post(
        "/v1/public-api-create-request",
        data={
            "customer_id": "test-cust",
            "experiment_name": "UCE test",
            "model_base_name": "test-model",
            "concept_forget": "cat",
            "concept_retain": "animal",
            "unlearning_algorithm": "UCE",
            "model_output_hf_id": "test/output",
        },
    )
    assert response.status_code == 200
    uuid = response.json()
    assert isinstance(uuid, str)
    assert len(uuid) == 36  # standard UUID length


def test_create_request_fade_no_zip(client: TestClient) -> None:
    """FADE with a non-zip upload must be rejected with 400."""
    response = client.post(
        "/v1/public-api-create-request",
        data={
            "customer_id": "test-cust",
            "experiment_name": "FADE test",
            "model_base_name": "test-model",
            "concept_forget": "cat",
            "concept_overwrite": "dog",
            "unlearning_algorithm": "FADE",
            "model_output_hf_id": "test/output",
        },
        files={"dataset": ("notazip.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400
    assert "zip" in response.json()["detail"].lower()


def test_create_request_fade_missing_forget_folder(client: TestClient) -> None:
    """Zip without a forget/ folder must be rejected with 400."""
    zip_bytes = _make_dataset_zip(include_forget=False, include_retain=True)
    response = client.post(
        "/v1/public-api-create-request",
        data={
            "customer_id": "test-cust",
            "experiment_name": "FADE test",
            "model_base_name": "test-model",
            "concept_forget": "cat",
            "concept_overwrite": "dog",
            "unlearning_algorithm": "FADE",
            "model_output_hf_id": "test/output",
        },
        files={"dataset": ("data.zip", zip_bytes, "application/zip")},
    )
    assert response.status_code == 400
    assert "forget" in response.json()["detail"].lower()


def test_create_request_fade_missing_retain_folder(client: TestClient) -> None:
    """Zip without a retain/ folder must be rejected with 400."""
    zip_bytes = _make_dataset_zip(include_forget=True, include_retain=False)
    response = client.post(
        "/v1/public-api-create-request",
        data={
            "customer_id": "test-cust",
            "experiment_name": "FADE test",
            "model_base_name": "test-model",
            "concept_forget": "cat",
            "concept_overwrite": "dog",
            "unlearning_algorithm": "FADE",
            "model_output_hf_id": "test/output",
        },
        files={"dataset": ("data.zip", zip_bytes, "application/zip")},
    )
    assert response.status_code == 400
    assert "retain" in response.json()["detail"].lower()


def test_create_request_fade_happy_path(client: TestClient) -> None:
    """Full FADE flow with a valid zip: returns UUID, DB has one record."""
    zip_bytes = _make_dataset_zip()
    response = client.post(
        "/v1/public-api-create-request",
        data={
            "customer_id": "test-cust",
            "experiment_name": "FADE happy",
            "model_base_name": "test-model",
            "concept_forget": "cat",
            "concept_overwrite": "dog",
            "unlearning_algorithm": "FADE",
            "model_output_hf_id": "test/output",
        },
        files={"dataset": ("data.zip", zip_bytes, "application/zip")},
    )
    assert response.status_code == 200
    uuid = response.json()
    assert isinstance(uuid, str) and len(uuid) == 36


def test_read_requests(client: TestClient, tmp_db: DatabaseLocalJson) -> None:
    """List requests for a customer, sorted newest-first."""
    # Create two UCE requests
    r1 = client.post(
        "/v1/public-api-create-request",
        data={
            "customer_id": "cust-A",
            "experiment_name": "exp-1",
            "model_base_name": "m",
            "concept_forget": "cat",
            "concept_retain": "animal",
            "unlearning_algorithm": "UCE",
            "model_output_hf_id": "o",
        },
    )
    r2 = client.post(
        "/v1/public-api-create-request",
        data={
            "customer_id": "cust-A",
            "experiment_name": "exp-2",
            "model_base_name": "m",
            "concept_forget": "dog",
            "concept_retain": "animal",
            "unlearning_algorithm": "UCE",
            "model_output_hf_id": "o",
        },
    )
    assert r1.status_code == 200
    assert r2.status_code == 200

    resp = client.get("/v1/public-api-read-requests", params={"customer_id": "cust-A"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2


def test_read_requests_empty(client: TestClient) -> None:
    """No requests returns an empty list."""
    resp = client.get("/v1/public-api-read-requests", params={"customer_id": "nobody"})
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_request_invalid_algorithm(client: TestClient) -> None:
    """Unknown algorithm with no dataset: automatic selection should fail."""
    response = client.post(
        "/v1/public-api-create-request",
        data={
            "customer_id": "test-cust",
            "experiment_name": "bad algo",
            "model_base_name": "m",
            "concept_forget": "cat",
            "unlearning_algorithm": "Automatic (recommended)",
            "model_output_hf_id": "o",
        },
    )
    # No image data and no concept_overwrite/concept_retain ->
    # infer_request raises HTTPException(400)
    assert response.status_code == 400


def test_create_request_uce_missing_concept_retain(client: TestClient) -> None:
    """UCE without concept_retain must be rejected."""
    response = client.post(
        "/v1/public-api-create-request",
        data={
            "customer_id": "test-cust",
            "experiment_name": "UCE bad",
            "model_base_name": "m",
            "concept_forget": "cat",
            "unlearning_algorithm": "UCE",
            "model_output_hf_id": "o",
        },
    )
    assert response.status_code == 400
    assert "retain" in response.json()["detail"].lower()
