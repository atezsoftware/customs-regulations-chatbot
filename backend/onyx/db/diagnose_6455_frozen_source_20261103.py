"""Read exactly one frozen 6455 derivative through the existing DEV diagnostic role."""

from __future__ import annotations

import argparse
import base64
import contextlib
import gzip
import hashlib
import io
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, cast

PACKAGE = "d2d0756e-e996-46cf-991b-aec286ed4aab"
ORIGINAL_SHA = "3f5ccdac09f51a64037e931deb0ccf1257ac941bd2c0d284ead3885e068d7366"
URL = (
    "https://www5.tbmm.gov.tr/tutanaklar/KANUNLAR_KARARLAR/"
    "kanuntbmmc097/kanuntbmmc097/kanuntbmmc09706455.pdf"
)
NAMESPACE = "customs-regulations-dev"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def remote(expected: str, manifest_sha: str) -> dict[str, Any]:
    # Imports use the deployed image, never the diagnostic branch's application code.
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import URL as DatabaseURL
    from sqlalchemy.orm import Session

    from onyx import __version__
    from onyx.file_store.file_store import (
        FileStore,
        S3BackedFileStore,
        get_default_file_store,
    )
    from onyx.regulatory.amendments.pdf_vision import load_pdf_extraction

    assert __version__ == expected
    assert os.environ.get("POSTGRES_DB") == NAMESPACE
    engine = create_engine(
        DatabaseURL.create(
            "postgresql+psycopg2",
            username=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            host=os.environ["POSTGRES_HOST"],
            port=int(os.environ.get("POSTGRES_PORT") or 5432),
            database=NAMESPACE,
        ),
        connect_args={
            "connect_timeout": 8,
            "options": "-c default_transaction_read_only=on -c search_path=public "
            "-c statement_timeout=15000 -c lock_timeout=1000 "
            "-c application_name=tariff_6455_frozen_readonly",
            **(
                {"sslmode": os.environ["POSTGRES_SSLMODE"]}
                if os.environ.get("POSTGRES_SSLMODE")
                else {}
            ),
        },
        pool_size=1,
        max_overflow=0,
    )
    try:
        with Session(engine) as session:
            identity = session.execute(
                text(
                    "SELECT current_database(),current_setting('transaction_read_only')"
                )
            ).one()
            assert identity[0] == NAMESPACE and identity[1] == "on"
            row = (
                session.execute(
                    text(
                        "SELECT p.id,p.document_set_id,p.environment,p.status,p.manifest_file_id,p.manifest_sha256,"
                        "p.input_file_id,p.request_hash,p.input_spec,p.created_by=(SELECT user_id FROM api_key WHERE id=5) AS owned_by_key5,"
                        "(SELECT count(*) FROM regulatory_document_intake i WHERE i.source_package_id=p.id) AS intakes "
                        "FROM amendment_source_package p WHERE p.id=:package AND p.document_set_id=15"
                    ),
                    {"package": PACKAGE},
                )
                .mappings()
                .one()
            )
            assert str(row["id"]) == PACKAGE and row["environment"] == "dev"
            assert row["owned_by_key5"] is True and row["intakes"] == 0
            assert row["status"] == "ready" and row["manifest_sha256"] == manifest_sha
            assert (
                row["request_hash"]
                == "f9351ceb17cf49a8af58fe91a8d72d906a4d81f0b2a9e6c64f16938341f40adb"
            )
            spec = row["input_spec"]
            assert spec["mime_type"] == "application/pdf" and spec["base_url"] == URL
            assets = (
                session.execute(
                    text(
                        "SELECT id,file_id,sha256,mime_type,byte_count,original_url,final_url,text_sha256 "
                        "FROM regulatory_source_asset WHERE package_id=:package"
                    ),
                    {"package": PACKAGE},
                )
                .mappings()
                .all()
            )
            assert len(assets) == 1
            asset = assets[0]
            assert asset["sha256"] == ORIGINAL_SHA and asset["byte_count"] == 396152
            assert asset["mime_type"] == "application/pdf"
            assert asset["original_url"] == asset["final_url"] == URL
            store = get_default_file_store()
            assert isinstance(store, S3BackedFileStore)
            allowed = {row["input_file_id"], row["manifest_file_id"], asset["file_id"]}
            payloads: dict[str, bytes] = {}

            class ScopedReadStore:
                def read_file(self, file_id: str) -> io.BytesIO:
                    assert file_id in allowed
                    if file_id not in payloads:
                        # Every FileRecord lookup uses this explicitly read-only session.
                        with store.read_file(file_id, db_session=session) as stream:
                            raw = stream.read(25 * 1024 * 1024 + 1)
                        assert len(raw) <= 25 * 1024 * 1024
                        payloads[file_id] = raw
                    return io.BytesIO(payloads[file_id])

            reader = ScopedReadStore()
            with reader.read_file(row["input_file_id"]) as stream:
                uploaded = stream.read()
            assert len(uploaded) == 396152 and sha(uploaded) == ORIGINAL_SHA
            assert (
                sha(json.dumps(spec, sort_keys=True).encode() + b"\0" + uploaded)
                == row["request_hash"]
            )
            with reader.read_file(asset["file_id"]) as stream:
                assert stream.read() == uploaded
            with reader.read_file(row["manifest_file_id"]) as stream:
                manifest_bytes = stream.read()
            assert sha(manifest_bytes) == manifest_sha
            manifest = json.loads(manifest_bytes)
            assert (
                manifest["status"] == "ready"
                and not manifest["issues"]
                and not manifest["links"]
            )
            assert len(manifest["assets"]) == 1
            prepared = manifest["assets"][0]
            assert prepared["sha256"] == ORIGINAL_SHA and prepared["page_count"] == 34
            assert prepared["mime_type"] == "application/pdf"
            assert prepared["original_url"] == prepared["final_url"] == URL
            reference = prepared["pdf_vision"]
            allowed.add(reference["file_id"])
            extraction = load_pdf_extraction(
                prepared, ORIGINAL_SHA, cast(FileStore, reader)
            )
            assert extraction is not None and extraction.page_count == 34
            assert not extraction.issues and extraction.model_snapshot is not None
            assert (
                sha(prepared["text"].encode())
                == reference["transcript_sha256"]
                == asset["text_sha256"]
            )
            raw = payloads[reference["file_id"]]
            assert sha(raw) == reference["sha256"]
            pages = sorted(
                {
                    e.locator.page
                    for e in extraction.elements
                    if e.extraction_method == "vision" and not e.aggregate
                }
            )
            assert pages == list(range(1, 35))
            return {
                "diagnostic": "6455_frozen_derivative_readonly",
                "revision": __version__,
                "database": NAMESPACE,
                "transaction_read_only": True,
                "package_id": PACKAGE,
                "manifest_sha256": manifest_sha,
                "source_sha256": ORIGINAL_SHA,
                "input_sha256": sha(uploaded),
                "source_asset_id": str(asset["id"]),
                "pdf_vision_reference": reference,
                "page_count": extraction.page_count,
                "element_count": len(extraction.elements),
                "model_snapshot": extraction.model_snapshot.model_dump(),
                "exact_derivative_gzip_base64": base64.b64encode(
                    gzip.compress(raw, mtime=0)
                ).decode(),
                "provider_calls": 0,
                "native_parse_calls": 0,
                "application_mutations": 0,
            }
    finally:
        engine.dispose()


def launch(expected: str, manifest_sha: str) -> None:
    pods = json.loads(
        subprocess.check_output(
            ["kubectl", "get", "pods", "-n", NAMESPACE, "-o", "json"], timeout=30
        )
    )["items"]
    targets = []
    for pod in pods:
        if (
            pod["status"]["phase"] != "Running"
            or pod["metadata"].get("deletionTimestamp")
            or pod["metadata"].get("labels", {}).get("app")
            != "dev-customs-regulations-background"
        ):
            continue
        statuses = {c["name"]: c for c in pod["status"].get("containerStatuses", [])}
        for container in pod["spec"]["containers"]:
            if container["image"].endswith(
                "customs-regulations-backend-lite-dev:" + expected
            ) and statuses.get(container["name"], {}).get("ready"):
                targets.append((pod["metadata"]["name"], container["name"]))
    assert targets
    pod, container = sorted(targets)[0]
    command = [
        "kubectl",
        "exec",
        "-i",
        pod,
        "-n",
        NAMESPACE,
        "-c",
        container,
        "--",
        "sh",
        "-ec",
        '. /vault/secrets/config; export PYTHONPATH=/app; exec python - --remote --expected-version "$1" --manifest-sha "$2"',
        "sh",
        expected,
        manifest_sha,
    ]
    result = subprocess.run(
        command,
        input=Path(__file__).read_text(),
        text=True,
        capture_output=True,
        timeout=180,
    )
    accepted = []
    for line in result.stdout.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if (
            record.get("diagnostic") == "6455_frozen_derivative_readonly"
            or "diagnostic_error_class" in record
        ):
            accepted.append(record)
    assert len(accepted) == 1
    payload = accepted[0].pop("exact_derivative_gzip_base64", None)
    if isinstance(payload, str):
        chunks = [payload[i : i + 20000] for i in range(0, len(payload), 20000)]
        accepted[0]["public_source_chunk_count"] = len(chunks)
        print(json.dumps(accepted[0]))
        for index, chunk in enumerate(chunks):
            print(
                json.dumps(
                    {
                        "frozen_derivative_chunk": index,
                        "total_chunks": len(chunks),
                        "gzip_base64": chunk,
                    }
                )
            )
    else:
        print(json.dumps(accepted[0]))
    if result.returncode or "diagnostic_error_class" in accepted[0]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--manifest-sha", required=True)
    args = parser.parse_args()
    assert re.fullmatch(r"[0-9a-f]{40}", args.expected_version)
    assert re.fullmatch(r"[0-9a-f]{64}", args.manifest_sha)
    if not args.remote:
        launch(args.expected_version, args.manifest_sha)
        return
    try:
        # Never print Vault data, provider prompts or raw exception strings.
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            logging.disable(logging.CRITICAL)
            report = remote(args.expected_version, args.manifest_sha)
        print(json.dumps(report))
    except Exception as error:
        print(
            json.dumps(
                {
                    "diagnostic_error_class": type(error).__name__,
                    "application_mutations": 0,
                }
            )
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
