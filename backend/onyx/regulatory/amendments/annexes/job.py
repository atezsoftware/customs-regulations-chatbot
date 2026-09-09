import hashlib
import io
import json
import time
from urllib.parse import urljoin
from uuid import UUID

from onyx.configs.constants import FileOrigin
from onyx.db.amendment_sources import (
    claim_source_package,
    finish_source_package,
    list_source_assets,
    mark_source_package_failed,
)
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import RegulatorySourceAsset
from onyx.file_store.file_store import FileStore, get_default_file_store
from onyx.regulatory.amendments.annexes.sources import (
    MAX_ASSET_BYTES,
    MAX_PACKAGE_SECONDS,
    DownloadedSource,
    acquire_source_package,
)
from onyx.regulatory.amendments.annexes.sources import (
    download_source_bounded as download_source,
)


def _read_verified_asset(store: FileStore, asset: RegulatorySourceAsset) -> bytes:
    with store.read_file(asset.file_id) as stream:
        content = stream.read(MAX_ASSET_BYTES + 1)
    if (
        len(content) != asset.byte_count
        or hashlib.sha256(content).hexdigest() != asset.sha256
    ):
        raise ValueError("Stored source integrity check failed")
    return content


def run_source_package(*, package_id: UUID, environment: str) -> None:
    with get_session_with_current_tenant() as session:
        claimed = claim_source_package(
            session, package_id=package_id, environment=environment
        )
        if claimed is None:
            return
        package, token = claimed
        spec = package.input_spec
        input_file_id = package.input_file_id
        previous_manifest_id = package.manifest_file_id
        existing_assets = list_source_assets(session, package_id)
    deadline = time.monotonic() + MAX_PACKAGE_SECONDS
    try:
        store = get_default_file_store()
        url = spec.get("url") or spec.get("base_url")
        mime_type = spec.get("mime_type")
        payload: bytes | None = None
        existing_by_hash = {asset.sha256: asset for asset in existing_assets}
        cached_by_url: dict[str, RegulatorySourceAsset] = {}
        for asset in existing_assets:
            for address in (asset.original_url, asset.final_url):
                if address:
                    cached_by_url[address] = asset
        if previous_manifest_id:
            with store.read_file(previous_manifest_id) as stream:
                previous_manifest = json.load(stream)
            for link in previous_manifest["links"]:
                target_hash = link.get("target_asset_hash")
                if link["kind"] == "url" and target_hash in existing_by_hash:
                    if link.get("final_url"):
                        cached_by_url[link["final_url"]] = existing_by_hash[target_hash]
                    parent = existing_by_hash[link["parent_asset_hash"]]
                    address = urljoin(
                        parent.final_url or "", link.get("original_url") or ""
                    )
                    if address:
                        cached_by_url[address] = existing_by_hash[target_hash]
            if previous_manifest["assets"]:
                root = existing_by_hash[previous_manifest["assets"][0]["sha256"]]
                payload = _read_verified_asset(store, root)
                mime_type = root.mime_type
                url = root.final_url or url
        if payload is None and input_file_id:
            with store.read_file(input_file_id) as stream:
                payload = stream.read(MAX_ASSET_BYTES + 1)

        def fetch(address: str) -> DownloadedSource:
            if cached := cached_by_url.get(address):
                return DownloadedSource(
                    _read_verified_asset(store, cached),
                    cached.mime_type,
                    cached.final_url or address,
                )
            return download_source(address, deadline=deadline)

        result = acquire_source_package(
            url=url,
            content=payload,
            mime_type=mime_type,
            display_name=spec.get("display_name", "source"),
            fetch=fetch,
        )
        stored_assets: list[RegulatorySourceAsset] = []
        for asset in result.assets:
            if asset.sha256 in existing_by_hash:
                continue
            file_id = store.save_file(
                io.BytesIO(asset.content),
                display_name=asset.display_name,
                file_origin=FileOrigin.OTHER,
                file_type=asset.mime_type,
            )
            text_bytes = asset.text.encode()
            text_file_id = (
                store.save_file(
                    io.BytesIO(text_bytes),
                    display_name="extracted.txt",
                    file_origin=FileOrigin.OTHER,
                    file_type="text/plain",
                )
                if text_bytes
                else None
            )
            stored_assets.append(
                RegulatorySourceAsset(
                    package_id=package_id,
                    sha256=asset.sha256,
                    file_id=file_id,
                    text_file_id=text_file_id,
                    text_sha256=hashlib.sha256(text_bytes).hexdigest()
                    if text_bytes
                    else None,
                    mime_type=asset.mime_type,
                    display_name=asset.display_name,
                    byte_count=len(asset.content),
                    original_url=asset.original_url,
                    final_url=asset.final_url,
                )
            )
        manifest = result.model_dump_json().encode()
        manifest_id = store.save_file(
            io.BytesIO(manifest),
            display_name="source-manifest.json",
            file_origin=FileOrigin.OTHER,
            file_type="application/json",
        )
        with get_session_with_current_tenant() as session:
            finish_source_package(
                session,
                package_id=package_id,
                environment=environment,
                lease_token=token,
                result=result,
                assets=stored_assets,
                manifest_file_id=manifest_id,
                manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            )
    except Exception:
        with get_session_with_current_tenant() as session:
            mark_source_package_failed(
                session,
                package_id=package_id,
                environment=environment,
                lease_token=token,
            )
        raise
