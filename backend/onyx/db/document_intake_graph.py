"""Bind ZIP document ownership to the retained, hash-verified source manifest."""

import hashlib
import json
import re
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import (
    AmendmentSourcePackage,
    RegulatoryDocumentIntakeFile,
    RegulatorySourceAsset,
)
from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.document_intake.source_graph import (
    ZIP_INPUT_KIND,
    IntakeGraph,
    parse_intake_graph,
)


def require_frozen_zip_intake_graph(
    session: Session,
    package: AmendmentSourcePackage,
    assets: Sequence[RegulatorySourceAsset],
    links: Sequence[SourceLink],
    *,
    intake_graph: IntakeGraph | None,
    manifest_bytes: bytes | None,
) -> IntakeGraph | None:
    if package.input_spec.get("kind") != ZIP_INPUT_KIND:
        if intake_graph is not None:
            raise ValueError("An intake ownership graph requires the ZIP input route")
        return None
    if (
        intake_graph is None
        or manifest_bytes is None
        or len(manifest_bytes) > 150 * 1024 * 1024
        or hashlib.sha256(manifest_bytes).hexdigest() != package.manifest_sha256
    ):
        raise ValueError(
            "The ZIP ownership graph must match its frozen source manifest"
        )
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("The ZIP source manifest is invalid")
    input_sha256 = package.input_spec.get("input_sha256")
    if (
        not isinstance(input_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", input_sha256) is None
        or manifest.get("input_sha256") != input_sha256
    ):
        raise ValueError("The ZIP source manifest does not match its original archive")
    raw_assets = manifest.get("assets")
    if (
        manifest.get("status") != "ready"
        or manifest.get("issues")
        or not isinstance(raw_assets, list)
        or len(raw_assets) != len(assets)
        or any(not isinstance(value, dict) for value in raw_assets)
        or any(
            not isinstance(value.get("sha256"), str)
            or not isinstance(value.get("mime_type"), str)
            for value in raw_assets
        )
        or {(value.get("sha256"), value.get("mime_type")) for value in raw_assets}
        != {(asset.sha256, asset.mime_type) for asset in assets}
    ):
        raise ValueError("The ZIP source manifest does not cover its document assets")
    raw_links = manifest.get("links")
    if not isinstance(raw_links, list):
        raise ValueError("The ZIP source graph is invalid")
    frozen_links = [SourceLink.model_validate(value) for value in raw_links]
    if frozen_links != list(links):
        raise ValueError("The ZIP source links changed before admission")
    fields = (
        RegulatorySourceAsset.id,
        RegulatorySourceAsset.package_id,
        RegulatorySourceAsset.sha256,
        RegulatorySourceAsset.mime_type,
        RegulatorySourceAsset.file_id,
        RegulatorySourceAsset.byte_count,
        RegulatorySourceAsset.text_file_id,
        RegulatorySourceAsset.text_sha256,
        RegulatorySourceAsset.original_url,
        RegulatorySourceAsset.final_url,
    )
    stored = {
        tuple(row)
        for row in session.execute(
            select(*fields).where(RegulatorySourceAsset.package_id == package.id)
        )
    }
    supplied = {
        (
            asset.id,
            asset.package_id,
            asset.sha256,
            asset.mime_type,
            asset.file_id,
            asset.byte_count,
            asset.text_file_id,
            asset.text_sha256,
            asset.original_url,
            asset.final_url,
        )
        for asset in assets
    }
    if stored != supplied:
        raise ValueError(
            "The stored ZIP source asset identity changed before admission"
        )
    frozen = parse_intake_graph(
        manifest,
        input_kind=package.input_spec.get("kind"),
        asset_hashes=[asset.sha256 for asset in assets],
        links=frozen_links,
    )
    if frozen != intake_graph:
        raise ValueError(
            "The ZIP ownership graph differs from its frozen source manifest"
        )
    return frozen


def require_frozen_zip_members(
    members: Sequence[RegulatoryDocumentIntakeFile],
    assets: Sequence[RegulatorySourceAsset],
    links: Sequence[SourceLink],
    graph: IntakeGraph,
) -> None:
    by_asset = {member.source_asset_id: member for member in members}
    by_hash = {asset.sha256: by_asset[asset.id] for asset in assets}
    roots = set(graph.root_hashes)
    for digest, member in by_hash.items():
        parent_hash = graph.primary_parent_by_hash.get(digest)
        expected_parent = by_hash[parent_hash].id if parent_hash else None
        expected_links = [
            link.model_dump(mode="json")
            for link in links
            if link.parent_asset_hash == digest or link.target_asset_hash == digest
        ]
        if (
            member.role != ("root" if digest in roots else "annex")
            or member.parent_document_id != expected_parent
            or member.relationships != expected_links
        ):
            raise ValueError("The admitted ZIP ownership or source links changed")
