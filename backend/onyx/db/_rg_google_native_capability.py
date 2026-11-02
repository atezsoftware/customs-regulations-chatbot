"""DEV-only provider capability probe; no DB/index writes or source modification.

Default operation is a model-metadata GET. An input path additionally permits a
countTokens POST; --embed is the separate explicit switch for an embedding call.
This script is prepared for execution inside the configured DEV backend pod.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account
from sqlalchemy import text

from onyx.configs.app_configs import POSTGRES_DB
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
from onyx.db.search_settings import get_current_search_settings
from onyx.natural_language_processing.utils import get_tokenizer
from onyx.regulatory.amendments.annexes import config
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import traced_llm_call
from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable
from shared_configs.contextvars import get_current_tenant_id
from shared_configs.enums import EmbeddingProvider

MODEL = "gemini-embedding-2"
DOCUMENTED_LIMIT = 8192


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_request(session: AuthorizedSession, method: str, url: str, body: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    # Do not render exception messages, headers, credentials or complete provider
    # responses. Status and shape are enough to inspect capability support.
    try:
        response = session.request(method, url, json=body, timeout=(10, 45), allow_redirects=False)
    except Exception as error:
        return {"transport_error_type": type(error).__name__}, None
    summary: dict[str, Any] = {"http_status": response.status_code}
    try:
        payload = response.json()
    except ValueError:
        summary["response_not_json"] = True
        return summary, None
    if not isinstance(payload, dict):
        summary["response_shape"] = type(payload).__name__
        return summary, None
    summary["response_keys"] = sorted(payload)
    if not 200 <= response.status_code < 300:
        error = payload.get("error")
        if isinstance(error, dict):
            summary["provider_error_status"] = error.get("status")
            summary["provider_error_code"] = error.get("code")
        return summary, None
    return summary, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-image-sha", required=True)
    parser.add_argument("--observed-image-sha", required=True)
    parser.add_argument("--input-path", type=Path, help="Exact final wire text, including the established embedding instruction wrapper")
    parser.add_argument("--expected-input-sha")
    parser.add_argument("--embed", action="store_true", help="Permit one synchronous embedContent request with autoTruncate=false")
    args = parser.parse_args()
    require(bool(re.fullmatch(r"[0-9a-f]{40}", args.expected_image_sha)), "Expected an exact image SHA")
    require(args.expected_image_sha == args.observed_image_sha, "Selected image does not match the expected SHA")
    require(POSTGRES_DB == "customs-regulations-dev" and config.REGULATORY_ANNEX_ENVIRONMENT == "dev", "Expected explicit DEV configuration")
    require(get_current_tenant_id() == "public", "Expected the original public tenant")
    require(not args.embed or args.input_path is not None, "Embedding requires an explicit exact input file")
    report: dict[str, Any] = {"observed_at": datetime.now(timezone.utc).isoformat(), "expected_image_sha": args.expected_image_sha, "documented_input_limit": DOCUMENTED_LIMIT}
    set_is_ee_based_on_env_variable()
    SqlEngine.init_engine(pool_size=1, max_overflow=0)
    with get_session_with_current_tenant() as db:
        db.execute(text("SET TRANSACTION READ ONLY"))
        require(db.execute(text("SELECT current_database()")).scalar_one() == "customs-regulations-dev", "Connected database is not DEV")
        settings = get_current_search_settings(db)
        require(settings.id == 11 and settings.model_name == MODEL and settings.provider_type == EmbeddingProvider.GOOGLE, "Active embedding provider/model changed")
        # Normal configured EE decryption, exactly as the maintained provider
        # adapter obtains its service-account JSON; no encryption overrides.
        raw_credential = settings.api_key
        require(bool(raw_credential), "Configured Google embedding credential is missing")
        identity = json.loads(raw_credential)
        require(isinstance(identity, dict), "Configured Google embedding credential is invalid")
        project = identity.get("project_id")
        location = identity.get("location") or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
        require(isinstance(project, str) and bool(re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", project)), "Configured Google project format is invalid")
        require(isinstance(location, str) and bool(re.fullmatch(r"[a-z][a-z0-9-]{0,62}", location)), "Configured Google location format is invalid")
        dimension = settings.final_embedding_dim
        tokenizer = get_tokenizer(model_name=settings.model_name, provider_type=settings.provider_type)
        report["provider"] = {"search_settings_id": settings.id, "model": MODEL, "transport": "same configured Enterprise v1 REST as google-genai enterprise=True", "dimension": dimension, "location": location}
        db.rollback()
    credential = service_account.Credentials.from_service_account_info(identity, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    # The host and paths mirror google-genai2.13.0's public Enterprise client.
    host = "aiplatform.googleapis.com" if location == "global" else f"aiplatform.{location}.rep.googleapis.com" if location in {"us", "eu"} else f"{location}-aiplatform.googleapis.com"
    model_path = f"publishers/google/models/{MODEL}"
    scoped_path = f"projects/{project}/locations/{location}/{model_path}"
    base = f"https://{host}/v1"
    logging.getLogger("google.auth").setLevel(logging.WARNING)
    with AuthorizedSession(credential) as session:
        status, model = safe_request(session, "GET", f"{base}/{model_path}")
        report["model_metadata_get"] = status
        if model is not None:
            report["model_metadata_get"]["capabilities"] = {key: model.get(key) for key in ("name", "displayName", "version", "versionId", "inputTokenLimit", "outputTokenLimit", "supportedGenerationMethods", "supportedActions") if key in model}
        if args.input_path is not None:
            require(bool(args.expected_input_sha and re.fullmatch(r"[0-9a-f]{64}", args.expected_input_sha)), "An exact input SHA is required")
            raw = args.input_path.read_bytes()
            require(0 < len(raw) <= 131072 and sha(raw) == args.expected_input_sha, "Embedding input size or SHA does not match")
            content = raw.decode("utf-8")
            require(content.strip() == content and "\x00" not in content, "Embedding wire input must be exact nonempty UTF-8 text")
            report["input"] = {"sha256": sha(raw), "bytes": len(raw), "local_tokenizer": type(tokenizer).__name__, "local_approximation_tokens": len(tokenizer.encode(content))}
            with traced_llm_call(flow=LLMFlow.EMBED_PASSAGE, model=MODEL, provider="google", extra_config={"operation": "native_count_tokens_capability_probe", "input_sha256": sha(raw)}):
                status, count = safe_request(session, "POST", f"{base}/{scoped_path}:countTokens", {"contents": [{"role": "user", "parts": [{"text": content}]}]})
            report["count_tokens"] = status
            if count is not None:
                report["count_tokens"]["native_total_tokens"] = count.get("totalTokens")
                report["count_tokens"]["raw_count_fields"] = {key: count[key] for key in ("totalTokens", "totalBillableCharacters") if key in count}
            if args.embed:
                body = {"content": {"parts": [{"text": content}]}, "embedContentConfig": {"outputDimensionality": dimension, "autoTruncate": False}}
                with traced_llm_call(flow=LLMFlow.EMBED_PASSAGE, model=MODEL, provider="google", extra_config={"operation": "native_no_truncation_canary", "input_sha256": sha(raw), "auto_truncate": "false"}):
                    status, result = safe_request(session, "POST", f"{base}/{scoped_path}:embedContent", body)
                report["embed_content"] = status
                report["embed_content"]["requested_auto_truncate"] = False
                if result is not None:
                    embedding = result.get("embedding")
                    values = embedding.get("values") if isinstance(embedding, dict) else None
                    require(isinstance(values, list) and len(values) == dimension and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values), "Provider returned an invalid vector shape")
                    usage = result.get("usageMetadata") or {}
                    stats = embedding.get("statistics") or {}
                    report["embed_content"].update({"vector_dimension": len(values), "finite_values": True, "vector_sha256": sha(json.dumps(values, separators=(",", ":")).encode()), "raw_truncated_present": "truncated" in result or "truncated" in stats, "raw_truncated": result.get("truncated", stats.get("truncated")), "native_processed_tokens": usage.get("promptTokenCount", stats.get("token_count")), "auto_truncate_false_accepted": True})
                    require(report["embed_content"]["raw_truncated"] is not True, "Provider unexpectedly truncated the canary input")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
