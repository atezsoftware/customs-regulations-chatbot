import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from scripts import prepare_regulatory_publication_baselines as cli

from onyx.db.enums import IndexModelStatus
from onyx.db.models import SearchSettings
from onyx.db.regulatory_publication_baseline import BaselineAuditInputs
from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.publication_baseline import observed_baseline_binding
from onyx.regulatory.writer_publication_models import WriterPublicationManifest
from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
    OwnedAuthority,
)
from tests.unit.onyx.regulatory.test_publication_baseline import baseline_case


@pytest.mark.parametrize("resume", [False, True])
def test_cli_audits_without_writes_and_resumes_only_frozen_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    inputs, evidence = baseline_case()
    owner = OwnedAuthority(inputs.file.id)
    binding = observed_baseline_binding(evidence, inputs.canonical)
    manifest = WriterPublicationManifest(
        id=uuid4(),
        scope=owner.scope,
        user_file_id=inputs.file.id,
        kind="baseline",
        index_state_sha256="a" * 64,
        canonical_before_sha256=publication_digest(
            [r.model_dump(mode="json") for r in inputs.canonical]
        ),
        indexes=[binding.index],
        bindings=[binding],
        previous_binding_ids=[],
        canonical_revisions={binding.id: uuid4()},
    )
    initial = BaselineAuditInputs(
        canonical=inputs.canonical,
        bindings=[],
        status="COMPLETED",
        gate_closed=resume,
        pending_manifest=resume,
        manifest=manifest if resume else None,
    )
    after = replace(
        initial,
        bindings=[binding],
        gate_closed=False,
        pending_manifest=False,
        manifest=None,
    )
    states = iter([initial, after] if resume else [initial])
    setting = SearchSettings(
        id=11,
        status=IndexModelStatus.PRESENT,
        index_name="physical-index",
        model_name="legacy",
        model_dim=2,
    )
    transport = MagicMock()
    client = transport.__enter__.return_value.publication_client.return_value
    client.indices.exists.return_value = True
    client.indices.get.return_value = {
        "physical-index": {"settings": {"index": {"uuid": "physical-uuid"}}}
    }
    authority = MagicMock()
    authority.acquire.return_value = owner.owner
    monkeypatch.setattr(cli.SqlEngine, "init_engine", lambda **_kw: None)
    monkeypatch.setattr(cli, "ElasticsearchClient", lambda: transport)
    monkeypatch.setattr(cli, "PublicationStore", lambda *_a: authority)
    monkeypatch.setattr(cli, "baseline_audit_settings", lambda *_a: [setting])
    monkeypatch.setattr(cli, "baseline_audit_inputs", lambda *_a: next(states))
    monkeypatch.setattr(
        cli,
        "read_file_inventory",
        lambda *_a: [
            evidence.model_copy(update={"observed_projection": binding.projection})
            if resume
            else evidence
        ],
    )
    monkeypatch.setattr(cli, "recover_owned_writer_before_next", lambda _owner: _owner)
    monkeypatch.setattr(
        "onyx.db.regulatory_writer_publication.pending_writer_manifest",
        lambda _owner: manifest,
    )
    output = tmp_path / "report.jsonl"
    args = [
        "baseline",
        "--expected-database",
        "customs-regulations-dev",
        "--expected-index-uuid",
        "physical-uuid",
        "--file-id",
        str(inputs.file.id),
        "--output",
        str(output),
    ]
    if resume:
        args.append("--apply")
    monkeypatch.setattr("sys.argv", args)
    cli.main()
    report, summary = [json.loads(line) for line in output.read_text().splitlines()]
    assert report["state"] == ("ready" if resume else "legacy")
    assert report["applied"] is resume
    assert summary["summary"] == {report["state"]: 1}
    if not resume:
        authority.acquire.assert_not_called()
