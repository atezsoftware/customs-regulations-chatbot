import re
import stat
from pathlib import Path

from packaging.utils import canonicalize_name

_BACKEND_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT = _BACKEND_ROOT.parent
_HEAVY_IMPORT_PACKAGES = {
    "markitdown",
    "onnxruntime",
    "pandas",
    "playwright",
    "pypdfium2",
    "unstructured",
    "unstructured-client",
}
_MODEL_SERVER_ONLY_PACKAGES = {"torch", "triton"}


def _requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==", line)
        if match is not None:
            names.add(canonicalize_name(match.group(1)))
    return names


def test_runtime_lock_excludes_import_and_browser_stack() -> None:
    requirements = _BACKEND_ROOT / "requirements"
    runtime_names = _requirement_names(requirements / "runtime.txt")
    full_names = _requirement_names(requirements / "default.txt")
    model_server_names = _requirement_names(requirements / "model_server.txt")

    assert runtime_names.isdisjoint(
        _HEAVY_IMPORT_PACKAGES | _MODEL_SERVER_ONLY_PACKAGES
    )
    assert _HEAVY_IMPORT_PACKAGES <= full_names
    assert _MODEL_SERVER_ONLY_PACKAGES <= model_server_names
    assert not any(name.startswith("nvidia-") for name in runtime_names)
    assert any(name.startswith("nvidia-") for name in model_server_names)


def test_runtime_lite_docker_target_is_independent_and_parser_free() -> None:
    full_dockerfile = (_BACKEND_ROOT / "Dockerfile").read_text(encoding="utf-8")
    lite_dockerfile = (_BACKEND_ROOT / "Dockerfile.runtime-lite").read_text(
        encoding="utf-8"
    )

    assert "runtime-lite" not in full_dockerfile
    assert "requirements/default.txt" not in lite_dockerfile
    assert "COPY --from=runtime-lite-builder" in lite_dockerfile
    assert "COPY --from=builder" not in lite_dockerfile
    assert "playwright install" not in lite_dockerfile
    assert 'DOCUMENT_IMPORT_ENABLED="false"' in lite_dockerfile
    assert 'io.regulatory.role="runtime-lite"' in lite_dockerfile
    assert 'io.regulatory.document-import="false"' in lite_dockerfile


def test_deployments_use_published_images_without_host_builds() -> None:
    compose_root = _REPO_ROOT / "deployment" / "docker_compose"
    prod_lite = (compose_root / "docker-compose.regulatory-prod-lite.yml").read_text(
        encoding="utf-8"
    )
    importer = (compose_root / "docker-compose.regulatory-importer.yml").read_text(
        encoding="utf-8"
    )
    external_infra = (
        compose_root / "docker-compose.regulatory-external-infra.yml"
    ).read_text(encoding="utf-8")
    publisher = (compose_root / "publish-regulatory-images.sh").read_text(
        encoding="utf-8"
    )
    importer_runner = (compose_root / "regulatory-import-run.sh").read_text(
        encoding="utf-8"
    )
    production_validator = (
        compose_root / "validate-regulatory-production.sh"
    ).read_text(encoding="utf-8")
    production_deployer = (compose_root / "regulatory-prod-lite-deploy.sh").read_text(
        encoding="utf-8"
    )

    assert prod_lite.count("build: !reset null") == 6
    assert prod_lite.count("pull_policy: always") == 6
    assert "${ONYX_BACKEND_LITE_IMAGE:?" in prod_lite
    assert "${ONYX_WEB_SERVER_IMAGE:?" in prod_lite
    assert "${ONYX_MODEL_SERVER_IMAGE:-registry.invalid/" in prod_lite
    assert "dockerfile:" not in prod_lite
    assert "onyx.main:app" in prod_lite
    assert "alembic upgrade" not in prod_lite
    assert 'DOCUMENT_IMPORT_ENABLED: "false"' in prod_lite
    assert "indexing_model_server: !reset null" in prod_lite
    assert 'profiles: ["indexing-model-server"]' in prod_lite
    assert "regulatory_cache_data:/data" in prod_lite
    assert "appendonly" in prod_lite
    assert prod_lite.count('ENABLE_PAID_ENTERPRISE_EDITION_FEATURES: "true"') == 2
    assert prod_lite.count('LICENSE_ENFORCEMENT_ENABLED: "false"') == 2
    assert external_infra.count("!reset null") == 7
    assert external_infra.count('profiles: ["local-infra"]') == 4
    assert "${ONYX_IMPORTER_IMAGE:?" in importer
    assert "build:" not in importer
    assert "pull_policy: always" in importer
    assert 'entrypoint: ["python", "-m", "scripts.regulatory_import"]' in importer
    assert "ports:" not in importer
    assert "depends_on:" not in importer
    assert 'ENABLE_PAID_ENTERPRISE_EDITION_FEATURES: "true"' in importer
    assert 'LICENSE_ENFORCEMENT_ENABLED: "false"' in importer

    assert '"$backend_dir/Dockerfile.runtime-lite" runtime-lite' in publisher
    assert '"$web_dir/Dockerfile"' in publisher
    assert '"$backend_dir/Dockerfile.model_server" final' in publisher
    assert '"$backend_dir/Dockerfile" runtime' in publisher
    assert "model_mode=${3:-cloud}" in publisher
    assert 'if [ "$model_mode" = "local" ]; then' in publisher
    assert 'docker push "$lite_tagged"' in publisher
    assert "REGULATORY_SOURCE_REVISION=$source_revision" in publisher
    assert "ONYX_BACKEND_LITE_IMAGE=$lite_digest_ref" in publisher
    assert "ONYX_WEB_SERVER_IMAGE=$web_digest_ref" in publisher
    assert "ONYX_MODEL_SERVER_IMAGE=%s\\n" in publisher
    assert "ONYX_IMPORTER_IMAGE=$importer_digest_ref" in publisher
    assert "rev-parse HEAD" in publisher
    assert "diff --quiet --no-ext-diff" in publisher
    assert "diff --cached --quiet --no-ext-diff" in publisher
    assert "ls-files --others --exclude-standard" in publisher
    assert '.["io.regulatory.role"] == "importer"' in importer_runner
    assert '.["io.regulatory.document-import"] == "true"' in importer_runner
    assert '.["org.opencontainers.image.revision"] == $revision' in importer_runner
    assert "compatibility alias" in production_validator
    assert "regulatory-prod-lite-preflight.sh" in production_validator
    for forbidden_runtime_module in ("docling", "nvidia", "torch", "triton"):
        assert f'"{forbidden_runtime_module}"' in production_deployer


def test_production_bundle_uses_a_physical_allowlist() -> None:
    builder = (
        _REPO_ROOT / "deployment" / "docker_compose" / "build-regulatory-prod-bundle.sh"
    )
    script = builder.read_text(encoding="utf-8")
    match = re.search(r"bundle_files='(?P<files>.*?)'\n\nfor file", script, re.DOTALL)
    assert match is not None
    members = set(match.group("files").splitlines())

    assert (
        "deployment/docker_compose/docker-compose.regulatory-prod-lite.yml" in members
    )
    assert "deployment/docker_compose/env.regulatory-prod.template" in members
    assert "deployment/docker_compose/regulatory-prod-lite-deploy.sh" in members
    assert not any("importer" in member for member in members)
    assert not any("publish-regulatory" in member for member in members)
    assert not any(member.startswith("backend/") for member in members)
    assert not any(member.startswith("web/") for member in members)
    assert "git diff --quiet --no-ext-diff" in script
    assert "git diff --cached --quiet --no-ext-diff" in script
    assert "git ls-files --others --exclude-standard" in script
    assert "[MODEL_DIGEST]" in script
    assert "model_mode=cloud" in script
    assert "printf 'model_mode=%s\\n'" in script
    assert builder.stat().st_mode & stat.S_IXUSR


def test_helm_cloud_models_overlay_removes_local_model_workloads() -> None:
    chart_root = _REPO_ROOT / "deployment" / "helm" / "charts" / "onyx"
    cloud_values = (chart_root / "values-cloud-models.yaml").read_text(encoding="utf-8")
    inference_service = (
        chart_root / "templates" / "inference-model-service.yaml"
    ).read_text(encoding="utf-8")
    indexing_service = (
        chart_root / "templates" / "indexing-model-service.yaml"
    ).read_text(encoding="utf-8")
    configmap = (chart_root / "templates" / "configmap.yaml").read_text(
        encoding="utf-8"
    )

    assert cloud_values.count("replicaCount: 0") == 2
    assert 'DISABLE_MODEL_SERVER: "true"' in cloud_values
    assert 'MODEL_SERVER_HOST: ""' in cloud_values
    assert 'MODEL_SERVER_PORT: ""' in cloud_values
    assert 'INDEXING_MODEL_SERVER_HOST: ""' in cloud_values
    assert 'INDEXING_MODEL_SERVER_PORT: ""' in cloud_values
    assert "controls model-server workloads only" in cloud_values
    assert "inferenceCapability.replicaCount" in inference_service
    assert "indexCapability.replicaCount" in indexing_service
    assert "inferenceCapability.replicaCount" in configmap
    assert "indexCapability.replicaCount" in configmap
