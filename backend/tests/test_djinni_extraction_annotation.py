import json
from pathlib import Path

import pytest

from app.eval import djinni_extraction_annotation as annotation
from app.eval.djinni_extraction_annotation import AnnotationError, compute_iaa, validate_annotation


def test_generated_or_model_annotation_is_rejected_before_document_loading(tmp_path: Path):
    artifact = tmp_path / "fake-human.json"
    artifact.write_text(json.dumps({
        "metadata": {
            "schema_version": 2,
            "artifact_type": "human_extraction_annotation",
            "human": False,
            "independent": False,
            "saw_system_output": True,
            "annotator_id": "model",
            "annotator_role": "A",
        },
        "profiles": {},
    }))
    with pytest.raises(AnnotationError, match="human must be True"):
        validate_annotation(artifact, "A")


def test_missing_annotator_identity_is_rejected_before_document_loading(tmp_path: Path):
    artifact = tmp_path / "anonymous.json"
    artifact.write_text(json.dumps({
        "metadata": {
            "schema_version": 2,
            "artifact_type": "human_extraction_annotation",
            "human": True,
            "independent": True,
            "saw_system_output": False,
            "annotator_id": "",
            "annotator_role": "A",
        },
        "profiles": {},
    }))
    with pytest.raises(AnnotationError, match="annotator_id is required"):
        validate_annotation(artifact, "A")


def test_one_to_one_iaa_and_conflict_detection(monkeypatch, tmp_path: Path):
    text = "Used Selenium in a test suite."
    docs = {"cv1": {"text": text, "sha256": "doc-hash", "domain": "QA", "length_stratum": "short"}}
    hashes = {"manifest_sha256": "m", "guidelines_sha256": "g", "taxonomy_sha256": "t"}
    monkeypatch.setattr(annotation, "_documents", lambda: docs)
    monkeypatch.setattr(annotation, "_hashes", lambda: hashes)
    monkeypatch.setattr(annotation, "_taxonomy_keys", lambda: ["selenium"])

    mention = {
        "start": 5,
        "end": 13,
        "raw_text": "Selenium",
        "canonical_skill": "selenium",
        "oov_name": None,
        "evidence_status": "used",
        "candidate_confirmed": True,
    }

    def payload(role: str, annotator_id: str, status: str) -> dict:
        changed = {**mention, "evidence_status": status}
        return {
            "metadata": {
                "schema_version": 2,
                "artifact_type": "human_extraction_annotation",
                "human": True,
                "independent": True,
                "saw_system_output": False,
                "annotator_id": annotator_id,
                "annotator_role": role,
                **hashes,
            },
            "profiles": {"cv1": {"document_sha256": "doc-hash", "completed": True, "mentions": [changed]}},
        }

    path_a, path_b = tmp_path / "a.json", tmp_path / "b.json"
    path_a.write_text(json.dumps(payload("A", "hoa", "used")))
    path_b.write_text(json.dumps(payload("B", "phuc", "listed")))
    metrics, conflicts = compute_iaa(path_a, path_b)
    assert metrics["span_exact"]["f1"] == 1.0
    assert metrics["span_overlap"]["f1"] == 1.0
    assert metrics["candidate_micro"]["f1"] == 1.0
    assert metrics["n_conflicts"] == 1
    assert conflicts[0]["kind"] == "aligned_disagreement"
