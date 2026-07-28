"""Regression guards for deterministic Helm deployment semantics."""

import os
from pathlib import Path
import subprocess

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = (
    "customs-regulations-core-codebuild.yaml",
    "customs-regulations-backend-codebuild.yaml",
    "customs-regulations-ui-codebuild.yaml",
)


def _workflow_step(workflow_name: str, step_name: str) -> dict:
    workflow_path = REPOSITORY_ROOT / ".github" / "workflows" / workflow_name
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["build-and-deploy"]["steps"]
    return next(step for step in steps if step.get("name") == step_name)


def _run_shell_step(
    workflow_name: str,
    step_name: str,
    *,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    step = _workflow_step(workflow_name, step_name)
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", step["run"]],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, **environment},
        capture_output=True,
        text=True,
        check=False,
    )


def _written_environment(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )


@pytest.mark.parametrize("workflow_name", WORKFLOWS)
def test_helm_four_is_pinned_to_legacy_client_side_apply(
    workflow_name: str,
) -> None:
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / workflow_name).read_text(
        encoding="utf-8"
    )

    assert workflow.count("uses: actions/checkout@v5") == 2
    assert workflow.count("uses: azure/setup-helm@v5.0.0") == 1
    assert workflow.count("version: v4.2.3") == 1
    assert workflow.count("helm upgrade --install") == 1
    assert workflow.count("helm rollback") == 2
    assert workflow.count("--server-side=false") == 3
    assert workflow.count("--wait=watcher") == 2
    assert workflow.count("helm status") == 1
    assert 'release_status=$(jq -r \'.info.status // empty\'' in workflow
    assert '[[ "$release_status" == pending-* ]]' in workflow
    assert "another operation" not in workflow
    assert "CURRENT_REV" not in workflow
    assert "PREV_REV" not in workflow
    assert workflow.count("helm history") == 2
    assert (
        workflow.count(
            'select(.status == "deployed" or .status == "superseded")'
        )
        == 2
    )
    assert workflow.count('if [[ -n "$rollback_revision" ]]') == 2
    assert workflow.count(
        "No successful revision exists; cleaning up failed first installation"
    ) == 2


@pytest.mark.parametrize("workflow_name", WORKFLOWS)
@pytest.mark.parametrize(
    ("event_name", "git_ref", "dispatch_environment", "expected_env"),
    (
        ("workflow_dispatch", "refs/heads/develop", "production", "v1"),
        ("workflow_dispatch", "refs/heads/release/v1", "test", "test-v1"),
        ("push", "refs/heads/develop", "production", "dev"),
        ("push", "refs/heads/release/v1", "dev", "v1"),
    ),
)
def test_deployment_target_uses_only_the_event_specific_source(
    workflow_name: str,
    event_name: str,
    git_ref: str,
    dispatch_environment: str,
    expected_env: str,
    tmp_path: Path,
) -> None:
    github_env = tmp_path / "github.env"
    result = _run_shell_step(
        workflow_name,
        "Creating Environment Variables",
        environment={
            "WORKFLOW_EVENT_NAME": event_name,
            "WORKFLOW_GIT_REF": git_ref,
            "WORKFLOW_GIT_SHA": "sha-for-test",
            "DISPATCH_ENVIRONMENT": dispatch_environment,
            "DISPATCH_ACTION": "deploy",
            "DISPATCH_IMAGE_TAG": "",
            "GITHUB_ENV": str(github_env),
        },
    )

    assert result.returncode == 0, result.stderr
    assert _written_environment(github_env)["env_x"] == expected_env


@pytest.mark.parametrize("workflow_name", WORKFLOWS)
@pytest.mark.parametrize(
    ("event_name", "git_ref", "dispatch_environment"),
    (
        ("push", "refs/heads/feature/not-deployable", "dev"),
        ("workflow_dispatch", "refs/heads/develop", "unknown"),
        ("schedule", "refs/heads/develop", "dev"),
    ),
)
def test_unknown_deployment_target_fails_fast(
    workflow_name: str,
    event_name: str,
    git_ref: str,
    dispatch_environment: str,
    tmp_path: Path,
) -> None:
    result = _run_shell_step(
        workflow_name,
        "Creating Environment Variables",
        environment={
            "WORKFLOW_EVENT_NAME": event_name,
            "WORKFLOW_GIT_REF": git_ref,
            "WORKFLOW_GIT_SHA": "sha-for-test",
            "DISPATCH_ENVIRONMENT": dispatch_environment,
            "DISPATCH_ACTION": "deploy",
            "DISPATCH_IMAGE_TAG": "",
            "GITHUB_ENV": str(tmp_path / "github.env"),
        },
    )

    assert result.returncode != 0
    assert "Unsupported deployment" in result.stderr


@pytest.mark.parametrize("workflow_name", WORKFLOWS)
def test_redeploy_requires_the_dispatch_image_tag_directly(
    workflow_name: str,
    tmp_path: Path,
) -> None:
    github_env = tmp_path / "github.env"
    creation = _run_shell_step(
        workflow_name,
        "Creating Environment Variables",
        environment={
            "WORKFLOW_EVENT_NAME": "workflow_dispatch",
            "WORKFLOW_GIT_REF": "refs/heads/develop",
            "WORKFLOW_GIT_SHA": "sha-must-not-be-used",
            "DISPATCH_ENVIRONMENT": "dev",
            "DISPATCH_ACTION": "redeploy",
            "DISPATCH_IMAGE_TAG": "",
            "GITHUB_ENV": str(github_env),
        },
    )
    validation = _run_shell_step(
        workflow_name,
        "Validate Redeploy Image",
        environment={"REDEPLOY_IMAGE_TAG": ""},
    )

    assert creation.returncode == 0, creation.stderr
    assert _written_environment(github_env)["IMAGE_TAG"] == ""
    assert validation.returncode != 0
    assert "image_tag input is required" in validation.stderr
