"""
Shared adversarial I-CARE fixture data for the backend route tests (test_api_icare.py)
and the Streamlit UI tests (test_ui.py).

The values are adversarial on purpose: every value is derived from the identity of the
entity (and task) it belongs to, so tests can recompute the expected value independently.
A scrambled entity/task mapping, a swapped parameter, or a flipped metric direction
produces a wrong number, not just "a number".
"""
import json
import os
from typing import Any, Dict, List

from vision_unlearning.benchmarks.I_care.configuration import mp_to_direction

TASKS: List[str] = ["breeds", "scenes", "people"]
ENTITIES: Dict[str, List[str]] = {
    "breeds": ["beagle", "collie", "poodle"],
    "scenes": ["airport", "bakery", "castle"],
    "people": ["Ada Lovelace", "Alan Turing", "Grace Hopper"],
}


def entity_value(task: str, entity_index: int) -> float:
    """Identity-derived per-entity metric value: unique per (task, entity)."""
    return TASKS.index(task) * 100.0 + entity_index + 0.25


def pair_value(emitter_index: int, receiver_index: int) -> float:
    """Identity-derived per-pair interference value: unique per (emitter, receiver)."""
    return emitter_index * 10.0 + receiver_index + 0.5


def build_interference_per_entity_data() -> Dict[str, List[Dict[str, Any]]]:
    """Per-task entity records, shaped like GET /v1/public-api-read-interference-per-entity-all."""
    return {
        task: [
            {
                "name": entity,
                "metric_emitter_average_clip_diff_20": entity_value(task, i),
            }
            for i, entity in enumerate(ENTITIES[task])
        ]
        for task in TASKS
    }


def write_interference_per_entity_files(assets_dir: str) -> None:
    """Write assets/interference_per_entity_{task}.json for all three tasks."""
    os.makedirs(assets_dir, exist_ok=True)
    for task, records in build_interference_per_entity_data().items():
        with open(os.path.join(assets_dir, f"interference_per_entity_{task}.json"), "w", encoding="utf-8") as f:
            json.dump(records, f)


def build_interference_matrix_data() -> Dict[str, Any]:
    """An InterferenceMatrix RT result for (sd1.4, people, uce, clip_diff), shaped like
    ResultTemplateInterferenceMatrix.compute()'s return value."""
    entities = ENTITIES["people"]
    return {
        "metadata": {
            "RT": "ResultTemplateInterferenceMatrix",
            "model": "sd1.4",
            "task": "people",
            "unlearning_algorithm": "uce",
            "interference_pair": "clip_diff",
            "_metric_key_name": "interference_pair",
            "metric_direction": mp_to_direction["clip_diff"],
        },
        "result": [
            {"emitter": emitter, **{receiver: pair_value(i, j) for j, receiver in enumerate(entities)}}
            for i, emitter in enumerate(entities)
        ],
    }


def write_interference_matrix_result(assets_dir: str) -> Dict[str, Any]:
    """Write the pre-computed InterferenceMatrix RT result to its exact local path.

    The file path must match ResultTemplateInterferenceMatrix._get_data_path_local()
    exactly -- that is the point: a route test only passes if the backend converts the
    GUI parameters to the correct backend literals and the RT serializes them to this path.
    """
    data = build_interference_matrix_data()
    result_dir = os.path.join(assets_dir, "results", "InterferenceMatrix")
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "sd1.4_people_uce_clip_diff.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data
