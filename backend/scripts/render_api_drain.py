"""Helm postrenderer preserving in-flight DEV API responses during rollouts."""

import sys
from typing import Any, cast

import yaml

DEV_NAMESPACE = "customs-regulations-dev"
DEV_API_DEPLOYMENT = "dev-customs-regulations-api-deployment"
DEV_API_CONTAINER = "dev-customs-regulations-api-container"
TERMINATION_GRACE_SECONDS = 900
ROLLOUT_TIMEOUT_SECONDS = 1020


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping at {path}")
    return cast(dict[str, Any], value)


def _extend_timeout(target: dict[str, Any], key: str, minimum: int) -> None:
    current = target.get(key, minimum)
    if type(current) is not int or current < 0:
        raise ValueError(f"Invalid existing {key}")
    target[key] = max(current, minimum)


def render_api_drain(manifests: str) -> str:
    documents = [
        _mapping(document, "document")
        for document in yaml.safe_load_all(manifests)
        if document is not None
    ]
    deployments: list[dict[str, Any]] = []
    for document in documents:
        if document.get("kind") != "Deployment":
            continue
        metadata = _mapping(document.get("metadata"), "Deployment.metadata")
        if metadata.get("name") != DEV_API_DEPLOYMENT:
            continue
        if (
            document.get("apiVersion") != "apps/v1"
            or metadata.get("namespace") != DEV_NAMESPACE
        ):
            raise ValueError("Unexpected DEV API deployment scope")
        deployments.append(document)
    if len(deployments) != 1:
        raise ValueError("Expected exactly one DEV API deployment")
    deployment = deployments[0]
    spec = _mapping(deployment.get("spec"), "Deployment.spec")
    template = _mapping(spec.get("template"), "Deployment.spec.template")
    pod = _mapping(template.get("spec"), "Deployment.spec.template.spec")
    containers = pod.get("containers")
    if not isinstance(containers, list) or not containers:
        raise ValueError("Expected DEV API containers")
    expected_containers = [
        container
        for value in containers
        if (container := _mapping(value, "container")).get("name") == DEV_API_CONTAINER
    ]
    if len(expected_containers) != 1:
        raise ValueError("Expected exactly one DEV API container")
    _extend_timeout(pod, "terminationGracePeriodSeconds", TERMINATION_GRACE_SECONDS)
    _extend_timeout(spec, "progressDeadlineSeconds", ROLLOUT_TIMEOUT_SECONDS)
    return yaml.safe_dump_all(documents, sort_keys=False)


def main() -> int:
    try:
        rendered = render_api_drain(sys.stdin.read())
    except (ValueError, yaml.YAMLError) as error:
        print(
            f"DEV API drain rendering refused: {type(error).__name__}", file=sys.stderr
        )
        return 1
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
