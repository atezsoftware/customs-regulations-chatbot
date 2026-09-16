import hashlib
import io
import json
import math
import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit
from uuid import UUID

from onyx.configs.constants import FileOrigin
from onyx.db.amendment_sources import (
    claim_source_package,
    extend_source_package_lease,
    finish_source_package,
    list_source_assets,
    mark_source_package_failed,
)
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import RegulatorySourceAsset
from onyx.file_store.file_store import FileStore, get_default_file_store
from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.amendments.annexes.source_limits import (
    SOURCE_PACKAGE_LEASE_MARGIN_SECONDS,
    source_preparation_seconds,
)
from onyx.regulatory.amendments.annexes.sources import (
    MAX_ASSET_BYTES,
    MAX_PACKAGE_SECONDS,
    DownloadedSource,
    acquire_source_package,
)
from onyx.regulatory.amendments.annexes.sources import (
    download_source_bounded as download_source,
)


@dataclass(frozen=True)
class _CachedSourceOccurrence:
    asset: RegulatorySourceAsset
    final_url: str


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
        previous_manifest_sha256 = (
            package.manifest_sha256 if previous_manifest_id else None
        )
        existing_assets = list_source_assets(session, package_id)
    started = time.monotonic()
    acquisition_deadline = started + MAX_PACKAGE_SECONDS
    try:
        store = get_default_file_store()
        url = spec.get("url") or spec.get("base_url")
        mime_type = spec.get("mime_type")
        payload: bytes | None = None
        existing_by_hash = {asset.sha256: asset for asset in existing_assets}
        cached_by_url: dict[str, _CachedSourceOccurrence] = {}
        previous_pdf_assets: dict[str, dict[str, object]] = {}
        legacy_pdf_assets: dict[str, dict[str, object]] = {}
        for asset in existing_assets:
            for address in (asset.original_url, asset.final_url):
                if address:
                    cached_by_url[address] = _CachedSourceOccurrence(
                        asset, asset.final_url or address
                    )
        if previous_manifest_id:
            from onyx.regulatory.amendments.pdf_vision import read_verified

            if not previous_manifest_sha256:
                raise ValueError("source_manifest_missing")
            previous_manifest = json.loads(
                read_verified(
                    store,
                    previous_manifest_id,
                    previous_manifest_sha256,
                    limit=150 * 1024 * 1024,
                )
            )
            previous_pdf_assets = {
                item["sha256"]: item
                for item in previous_manifest["assets"]
                if item.get("pdf_vision")
            }
            legacy_pdf_assets = {
                item["sha256"]: item
                for item in previous_manifest["assets"]
                if item["mime_type"] == "application/pdf" and not item.get("pdf_vision")
            }
            if legacy_pdf_assets and previous_pdf_assets:
                raise ValueError("source_pdf_contract_mixed")
            for raw_link in previous_manifest["links"]:
                link = SourceLink.model_validate(raw_link)
                if (
                    link.kind != "url"
                    or link.target_asset_hash not in existing_by_hash
                    or not link.final_url
                ):
                    continue
                occurrence = _CachedSourceOccurrence(
                    existing_by_hash[link.target_asset_hash], link.final_url
                )
                cached_by_url[link.final_url] = occurrence
                address = link.requested_url
                if not address and link.parent_url:
                    address = urljoin(link.parent_url, link.original_url or "")
                if not address and urlsplit(link.original_url or "").scheme in (
                    "http",
                    "https",
                ):
                    address = link.original_url
                if address:
                    cached_by_url[address] = occurrence
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
                    _read_verified_asset(store, cached.asset),
                    cached.asset.mime_type,
                    cached.final_url,
                )
            return download_source(address, deadline=acquisition_deadline)

        result = acquire_source_package(
            url=url,
            content=payload,
            mime_type=mime_type,
            display_name=spec.get("display_name", "source"),
            fetch=fetch,
        )
        pending_pdfs = [
            asset
            for asset in result.assets
            if asset.mime_type == "application/pdf"
            and asset.sha256 not in previous_pdf_assets
            and not legacy_pdf_assets
        ]
        pending_images = [
            asset
            for asset in result.assets
            if asset.mime_type.startswith("image/")
            and not (
                (existing := existing_by_hash.get(asset.sha256))
                and existing.text_file_id
                and existing.text_sha256
            )
        ]
        if any(asset.sha256 in existing_by_hash for asset in pending_images):
            raise ValueError("image_source_requires_new_preparation")
        if any(asset.page_count is None for asset in pending_pdfs):
            raise ValueError("pdf_page_count_missing")
        if any(asset.page_count is None for asset in pending_images):
            raise ValueError("image_page_count_missing")
        deadline = started + source_preparation_seconds(
            sum(asset.page_count or 0 for asset in [*pending_pdfs, *pending_images])
        )
        if time.monotonic() >= deadline:
            raise TimeoutError("source_preparation_deadline")
        with get_session_with_current_tenant() as session:
            if not extend_source_package_lease(
                session,
                package_id=package_id,
                environment=environment,
                lease_token=token,
                lease_seconds=math.ceil(deadline - time.monotonic())
                + SOURCE_PACKAGE_LEASE_MARGIN_SECONDS,
            ):
                raise RuntimeError("source_preparation_lease_lost")
        if legacy_pdf_assets:
            # A retry completes the frozen package; it never partially upgrades its PDFs.
            result.assets = [
                asset.model_copy(
                    update={"text": legacy_pdf_assets[asset.sha256]["text"]}
                )
                if asset.sha256 in legacy_pdf_assets
                else asset
                for asset in result.assets
            ]
        elif any(asset.mime_type == "application/pdf" for asset in result.assets):
            from onyx.llm.factory import get_default_llm_with_vision
            from onyx.regulatory.amendments.pdf_vision import (
                prepare_pdf_source,
                reuse_pdf_source,
            )

            vision = None
            if any(
                asset.mime_type == "application/pdf"
                and asset.sha256 not in previous_pdf_assets
                for asset in result.assets
            ):
                vision = get_default_llm_with_vision(temperature=0)
            prepared = []
            for asset in result.assets:
                if time.monotonic() >= deadline:
                    raise TimeoutError("source_preparation_deadline")
                if asset.mime_type != "application/pdf":
                    prepared.append(asset)
                elif asset.sha256 in previous_pdf_assets:
                    prepared.append(
                        reuse_pdf_source(
                            asset, previous_pdf_assets[asset.sha256], store
                        )
                    )
                else:
                    prepared.append(
                        prepare_pdf_source(
                            asset, store=store, llm=vision, deadline=deadline
                        )
                    )
            result.assets = prepared
        if any(asset.mime_type.startswith("image/") for asset in result.assets):
            from onyx.llm.factory import get_default_llm_with_vision
            from onyx.regulatory.amendments.annexes.source_images import (
                prepare_image_source,
                reuse_image_source,
            )

            image_model = get_default_llm_with_vision() if pending_images else None
            prepared_images = []
            for asset in result.assets:
                if time.monotonic() >= deadline:
                    raise TimeoutError("source_preparation_deadline")
                if not asset.mime_type.startswith("image/"):
                    prepared_images.append(asset)
                    continue
                existing = existing_by_hash.get(asset.sha256)
                if existing and existing.text_file_id and existing.text_sha256:
                    prepared_images.append(
                        reuse_image_source(
                            asset,
                            store=store,
                            text_file_id=existing.text_file_id,
                            text_sha256=existing.text_sha256,
                        )
                    )
                else:
                    prepared_images.append(
                        prepare_image_source(asset, llm=image_model, deadline=deadline)
                    )
            result.assets = prepared_images
        stored_assets: list[RegulatorySourceAsset] = []
        for asset in result.assets:
            if time.monotonic() >= deadline:
                raise TimeoutError("source_preparation_deadline")
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
        if time.monotonic() >= deadline:
            raise TimeoutError("source_preparation_deadline")
        manifest = result.model_dump_json().encode()
        manifest_id = store.save_file(
            io.BytesIO(manifest),
            display_name="source-manifest.json",
            file_origin=FileOrigin.OTHER,
            file_type="application/json",
        )
        with get_session_with_current_tenant() as session:
            if time.monotonic() >= deadline:
                raise TimeoutError("source_preparation_deadline")
            if not finish_source_package(
                session,
                package_id=package_id,
                environment=environment,
                lease_token=token,
                result=result,
                assets=stored_assets,
                manifest_file_id=manifest_id,
                manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            ):
                raise RuntimeError("source_preparation_lease_lost")
    except Exception as error:
        with get_session_with_current_tenant() as session:
            mark_source_package_failed(
                session,
                package_id=package_id,
                environment=environment,
                lease_token=token,
                failure=error,
            )
        raise
