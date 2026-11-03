"""Validate the DEV-only drain change against Helm's multi-resource output."""

import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from scripts.render_api_drain import render_api_drain


def _deployment() -> dict[str, Any]:
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "dev-customs-regulations-api-deployment",
            "namespace": "customs-regulations-dev",
        },
        "spec": {
            "replicas": 2,
            "strategy": {
                "type": "RollingUpdate",
                "rollingUpdate": {"maxSurge": "100%", "maxUnavailable": 0},
            },
            "template": {
                "metadata": {
                    "annotations": {"vault.hashicorp.com/agent-inject": "true"}
                },
                "spec": {
                    "serviceAccountName": "dev-customs-regulations-api",
                    "containers": [
                        {
                            "name": "dev-customs-regulations-api-container",
                            "image": "example.invalid/backend:current",
                            "args": [
                                "bash",
                                "-c",
                                "source /vault/secrets/config && alembic upgrade head "
                                "&& exec uvicorn onyx.main:app --host 0.0.0.0 --port 8080",
                            ],
                        },
                        {"name": "sidecar", "image": "example.invalid/sidecar:current"},
                    ],
                },
            },
        },
    }


def test_multi_document_render_changes_only_expected_deployment_timeouts() -> None:
    api = _deployment()
    background = copy.deepcopy(api)
    background["metadata"]["name"] = "dev-customs-regulations-background-deployment"
    service = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "api"}}
    hook = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": "hook", "annotations": {"helm.sh/hook": "post-install"}},
    }
    documents: list[dict[str, Any]] = [service, background, api, hook]
    expected = copy.deepcopy(documents)
    expected[2]["spec"]["progressDeadlineSeconds"] = 1020
    expected[2]["spec"]["template"]["spec"]["terminationGracePeriodSeconds"] = 900
    rendered = render_api_drain(yaml.safe_dump_all(documents))
    assert list(yaml.safe_load_all(rendered)) == expected
    assert render_api_drain(rendered) == rendered


def test_render_preserves_larger_existing_drain_limits() -> None:
    deployment = _deployment()
    deployment["spec"]["progressDeadlineSeconds"] = 1800
    deployment["spec"]["template"]["spec"]["terminationGracePeriodSeconds"] = 1200
    assert yaml.safe_load(render_api_drain(yaml.safe_dump(deployment))) == deployment


@pytest.mark.parametrize("scope", ["missing", "duplicate", "namespace", "container"])
def test_render_refuses_missing_ambiguous_or_out_of_scope_api(scope: str) -> None:
    deployment = _deployment()
    documents = [deployment]
    if scope == "missing":
        documents = []
    elif scope == "duplicate":
        documents.append(copy.deepcopy(deployment))
    elif scope == "namespace":
        deployment["metadata"]["namespace"] = "customs-regulations-prod"
    else:
        deployment["spec"]["template"]["spec"]["containers"] = [
            {"name": "another-container"}
        ]
    with pytest.raises(ValueError):
        render_api_drain(yaml.safe_dump_all(documents))


@pytest.mark.parametrize("invalid", [None, True, "900", -1])
def test_render_refuses_invalid_existing_drain_limit(invalid: object) -> None:
    deployment = _deployment()
    deployment["spec"]["template"]["spec"]["terminationGracePeriodSeconds"] = invalid
    with pytest.raises(ValueError, match="Invalid existing"):
        render_api_drain(yaml.safe_dump(deployment))


@pytest.mark.parametrize(
    "manifests", ["kind: [", "---\n- invalid\n", "kind: Deployment\n"]
)
def test_cli_failure_emits_no_partial_manifest(manifests: str) -> None:
    script = Path(__file__).resolve().parents[3] / "scripts" / "render_api_drain.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        input=manifests,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert "DEV API drain rendering refused" in result.stderr


@pytest.mark.parametrize(
    "environment,app,protected",
    [
        ("dev", "customs-regulations-api", True),
        ("dev", "customs-regulations-background", False),
        ("test-v1", "customs-regulations-api", False),
        ("v1", "customs-regulations-api", False),
    ],
)
def test_deploy_function_drains_only_dev_api(
    tmp_path: Path, environment: str, app: str, protected: bool
) -> None:
    root = Path(__file__).resolve().parents[4]
    workflow = yaml.safe_load(
        (
            root / ".github/workflows/customs-regulations-backend-lite-codebuild.yaml"
        ).read_text()
    )
    step = next(
        step
        for step in workflow["jobs"]["build-and-deploy"]["steps"]
        if step["name"] == "Deploy api and background with Helm"
    )
    function = step["run"].split("\nverify_benchmark_worker()", 1)[0]
    commands = {
        "helm": """#!/bin/bash
printf '%s\\n' "helm|$*|plugins=${HELM_PLUGINS:-unset}" >> "$CALL_LOG"
if [[ "$1" == status ]]; then exit 1; fi
""",
        "kubectl": """#!/bin/bash
printf '%s\\n' "kubectl|$*" >> "$CALL_LOG"
if [[ "$1 $2" == 'get pods' ]]; then
  printf '%s\\n' '{"items":[{"metadata":{"name":"old","deletionTimestamp":"now"}}]}'
elif [[ "$1 $2" == 'get deployment' ]]; then
  printf '%s\\n' '{"spec":{"template":{"spec":{"containers":[{"image":"api:current"}]}}}}'
fi
""",
        "jq": """#!/bin/bash
cat >/dev/null
if [[ "$1" == -r ]]; then printf '%s\\n' old; fi
""",
    }
    for name, command in commands.items():
        executable = tmp_path / name
        executable.write_text(command)
        executable.chmod(0o755)
    log = tmp_path / "calls.log"
    result = subprocess.run(
        [
            "bash",
            "-c",
            function
            + '\ndeploy_app "$TEST_APP" fixture.yaml false ""\n'
            + 'printf "%s\\n" "after|$HELM_PLUGINS" >> "$CALL_LOG"',
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "CALL_LOG": str(log),
            "HELM_PLUGINS": "/existing/plugins",
            "TEST_APP": app,
            "env_x": environment,
            "namespace": "fixture",
            "space_x": environment,
            "IMAGE_TAG": "current",
        },
        check=False,
    )
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    upgrade = next(call for call in calls if call.startswith("helm|upgrade"))
    rollout = next(call for call in calls if call.startswith("kubectl|rollout"))
    assert ("--post-renderer dev-api-drain" in upgrade) is protected
    expected_timeout = "1020s" if protected else "600s"
    assert f"--timeout {expected_timeout}" in upgrade
    assert f"--timeout={expected_timeout}" in rollout
    assert any("--grace-period=0 --force" in call for call in calls) is not protected
    assert calls[-1] == "after|/existing/plugins"
