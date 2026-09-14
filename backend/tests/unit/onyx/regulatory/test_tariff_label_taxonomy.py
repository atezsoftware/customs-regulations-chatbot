import json
import os
import subprocess
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path

from onyx.regulatory.indexing_jobs.vertex_batch import build_vertex_jsonl
from onyx.regulatory.labeling.api_models import TaxonomyCreate
from onyx.regulatory.labeling.defaults import load_default_taxonomy
from onyx.regulatory.labeling.provider import (
    TaxonomyDefinition,
    build_labeling_request,
    validate_labeling_response,
)

TAXONOMY_PATH = (
    Path(__file__).resolve().parents[5]
    / "deployment/labeling/tariff-regulatory-intelligence-chunk-labels-v1.json"
)


def test_full_tariff_vocabulary_fits_existing_upload_and_batch_contracts() -> None:
    uploaded = TaxonomyCreate.model_validate(load_default_taxonomy().model_dump())
    taxonomy = TaxonomyDefinition.model_validate(uploaded.model_dump())
    assert len(taxonomy.labels) == 165
    families = Counter(label.id.split(".")[0] for label in taxonomy.labels)
    assert families == {
        "SUB": 90,
        "EFF": 33,
        "ANX": 30,
        "SEC": 12,
    }
    # Independent extraction of the user's supplied sections 2.4.5–2.4.8.
    assert (
        sha256(
            ("\n".join(sorted(label.id for label in taxonomy.labels)) + "\n").encode()
        ).hexdigest()
        == "466468adb3d2ab81302c08939ac7c2c733a8e376b1f835c62d5171072343db59"
    )
    assert uploaded.model_dump() == json.loads(TAXONOMY_PATH.read_text())
    request = build_labeling_request(
        chunk_id="existing-canonical-chunk",
        text="İthalatçı, ithalat işlemi için izin belgesini ibraz eder.",
        context="Bu parça mevcut dosya chunkıdır.",
        taxonomy=taxonomy,
        source_hash="a" * 64,
    )
    wire = json.loads(build_vertex_jsonl([request]))
    schema = wire["request"]["generationConfig"]["responseJsonSchema"]
    assert schema["properties"]["labels"]["items"]["properties"]["label_id"][
        "enum"
    ] == [label.id for label in taxonomy.labels]
    assert json.loads(request.prompt)["taxonomy"] == uploaded.model_dump()


def test_tariff_ids_across_families_retain_the_existing_evidence_output_shape() -> None:
    taxonomy = TaxonomyDefinition.model_validate_json(TAXONOMY_PATH.read_text())
    text = "İthalatçı, ithalat işlemi için izin belgesini ibraz eder."
    labels = [
        "SUB.TRD.IMPORT_CONTROL",
        "EFF.DOCUMENT",
    ]
    outcome = validate_labeling_response(
        json.dumps(
            {
                "labels": [
                    {"label_id": label, "evidence_quote": text} for label in labels
                ],
                "abstained": False,
            }
        ),
        text=text,
        taxonomy=taxonomy,
    )
    assert [assignment.label_id for assignment in outcome.labels] == labels
    assert all(
        set(assignment.model_dump()) == {"label_id", "evidence_quote"}
        for assignment in outcome.labels
    )


def test_bundled_labels_are_available_outside_the_repository_working_directory(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from onyx.regulatory.labeling.defaults import load_default_taxonomy; "
            "print(load_default_taxonomy().model_dump_json())",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[4]),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == json.loads(TAXONOMY_PATH.read_text())


def test_original_migration_seed_remains_immutable() -> None:
    legacy_path = TAXONOMY_PATH.with_name("tariff-regulatory-intelligence-v2.1.json")
    legacy = TaxonomyDefinition.model_validate_json(legacy_path.read_text())
    assert legacy.version_hash == (
        "5a89e4d393c2974a900e15bb57914814633e70ce65b0f5f39f02b7d24b7b50bd"
    )
    bundled = (
        Path(__file__).resolve().parents[4]
        / "onyx/regulatory/labeling/data/tariff-regulatory-intelligence-v2.1.json"
    )
    assert json.loads(bundled.read_text()) == legacy.model_dump()
