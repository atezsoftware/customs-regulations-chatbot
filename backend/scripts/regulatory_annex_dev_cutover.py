"""Bounded runner entrypoint for the existing DEV backend-lite deployment."""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

NAMESPACE = "customs-regulations-dev"
REPOSITORY = "255114580789.dkr.ecr.eu-central-1.amazonaws.com/customs-regulations-backend-lite-dev"
APPS = ("api", "background")
STATE = "regulatory-annex-dev-cutover"
PROBE = "regulatory-annex-dev-cutover-probe"
# Includes the bounded calibration record plus native/runtime evidence.
MAX_ACCEPTANCE_REPORT_BYTES = 400_000


class CutoverRefusal(RuntimeError):
    """A fixed diagnostic that is safe to expose in runner logs."""


def validate_scope(environment: str, ref: str, sha: str) -> None:
    if (
        environment != "dev"
        or ref != "refs/heads/develop"
        or not re.fullmatch("[0-9a-f]{40}", sha)
    ):
        raise ValueError("DEV develop exact SHA required")


def prepare(driver: "Driver") -> None:
    driver.validate_target()
    if driver.continue_protected_release():
        driver.record_installing()
        return
    driver.inventory()
    driver.create_probe()
    driver.inspect_indices()
    driver.record_started()
    driver.stop_api()
    driver.drain_background()
    driver.stop_background()
    driver.assert_no_writers()
    driver.block_indices()
    driver.record_blocked()
    driver.assert_no_writers()
    driver.unblock_indices()
    driver.record_installing()


def release(driver: "Driver") -> None:
    driver.validate_target()
    driver.load_state()
    driver.verify_runtime()
    driver.record_released()
    driver.delete_probe()


def emit_acceptance_report(stdout: str, phase: str, sha: str) -> None:
    """Retain only the fixed probe evidence, including a failed calibration verdict."""
    keys = {
        "phase",
        "status",
        "release_sha_metadata",
        "configuration",
        "native",
        "calibration",
        "canary",
        "retained",
        "reused_completed_run",
        "database",
        "indices",
        "contextual",
        "index_name",
        "contextual_enabled",
        "default_provider",
        "default_model",
        "vision_provider",
        "vision_model",
        "native_machine",
        "pdf_pages",
        "module_sha256",
        "fixture_sha256",
        "render_sha256",
        "dependencies",
        "source_fetches",
        "parser_limits_relaxed",
        "planned_cases",
        "cases",
        "format",
        "proposed_value",
        "expected_supported",
        "original_value",
        "raw_transcription",
        "supported",
        "rationale",
        "input_sha256",
        "failure",
        "attempt_count",
        "http_request_count",
        "attempt_count_complete",
        "model_snapshot",
        "model_provider",
        "model_name",
        "database_read_only",
        "fixture_verified",
        "rationale_truncated",
        "release_sha",
        "run_id",
        "user_id",
        "file_id",
        "document_set_id",
        "pat_id",
        "package_id",
        "batch_id",
        "review_id",
        "chat_ids",
        "markdown_file_ids",
        "created_at",
        "evidence",
        "physical_indices",
        "source_assets",
        "review_sha256",
        "approval_count",
        "publication_generation",
        "canonical_changes",
        "context_consumers",
        "embeddings",
        "exact_vector_reuses",
        "historical_projections",
        "retired_projections",
        "total_projections",
        "cleanup_complete",
        "creation_intents",
        "kind",
        "marker",
        "artifact_id",
        "retained_objects",
        "id",
        "index_uuid",
        "vision_roles",
        "side",
        "position",
        "table_role",
        "source_sha256",
        "locator_sha256",
        "source_file_id",
        "original_position",
        "original_locator",
        "original_locator_sha256",
        "view_position",
        "view_sha256",
        "view_locator_sha256",
        "ordinary_markdown_upload_index_attachment_chat",
        "acceptance_passed",
        "retained_tombstones",
        "cleanup_live_projections",
        "retained_source_scope",
        "cleanup_failure",
        "chat_cleanup_failure",
        "token_cleanup_failure",
        "chat_2026-09-09",
        "chat_2026-09-10",
    }

    def sanitize(value: Any, parent: str = "", depth: int = 0) -> Any:
        if depth > 8:
            raise CutoverRefusal("acceptance_report_depth_exceeded")
        if isinstance(value, dict):
            if len(value) > 100:
                raise CutoverRefusal("acceptance_report_mapping_exceeded")
            if parent in {"module_sha256", "fixture_sha256"}:
                return {
                    key: item
                    for key, item in value.items()
                    if re.fullmatch(r"[a-zA-Z0-9_.-]{1,160}", key)
                    and isinstance(item, str)
                    and re.fullmatch(r"[0-9a-f]{64}", item)
                }
            if parent == "dependencies":
                return {
                    key: item
                    for key, item in value.items()
                    if key in {"pypdfium2", "pillow", "openpyxl", "python-docx"}
                    and isinstance(item, str)
                    and re.fullmatch(r"[a-zA-Z0-9.+-]{1,40}", item)
                }
            return {
                key: sanitize(item, key, depth + 1)
                for key, item in value.items()
                if key in keys
            }
        if isinstance(value, list):
            if len(value) > (512 if parent == "retained_objects" else 100):
                raise CutoverRefusal("acceptance_report_list_exceeded")
            return [sanitize(item, parent, depth + 1) for item in value]
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str) and len(value) <= 4000:
            return value
        raise CutoverRefusal("acceptance_report_value_refused")

    reports = []
    for line in stdout.splitlines():
        if (
            not line.startswith("{")
            or len(line.encode("utf-8")) > MAX_ACCEPTANCE_REPORT_BYTES
        ):
            continue
        try:
            report = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(report, dict)
            and report.get("phase") == phase
            and report.get("release_sha_metadata") == sha
        ):
            reports.append(report)
    if len(reports) != 1 or reports[0].get("status") not in {"passed", "failed"}:
        raise CutoverRefusal("fixed_acceptance_report_required")
    print(json.dumps(sanitize(reports[0]), sort_keys=True), flush=True)
    if reports[0]["status"] != "passed":
        raise CutoverRefusal("fixed_acceptance_probe_failed")


class Driver:
    def __init__(self, sha: str) -> None:
        self.sha = sha
        self.indices: dict[str, str] = {}
        self.container = ""
        self.scope: dict[str, Any] = {}
        self.writer_nodes: set[str] = set()

    def command(
        self,
        args: list[str],
        stdin: str | None = None,
        timeout: int = 900,
        *,
        acceptance_phase: str | None = None,
    ) -> str:
        result = subprocess.run(
            args, input=stdin, text=True, capture_output=True, timeout=timeout
        )
        if acceptance_phase is not None:
            if acceptance_phase not in {"preflight", "canary"}:
                raise CutoverRefusal("fixed_acceptance_phase_required")
            emit_acceptance_report(result.stdout, acceptance_phase, self.sha)
        if result.returncode:
            # kubectl/provider errors can contain credentials or source text.
            raise CutoverRefusal(f"command_failed:{args[0]}")
        return result.stdout.strip()

    def kubectl(self, *args: str, stdin: str | None = None) -> str:
        return self.command(["kubectl", "--namespace", NAMESPACE, *args], stdin)

    def get(self, resource: str, name: str | None = None) -> Any:
        args = ["get", resource]
        if name:
            args.append(name)
        return json.loads(self.kubectl(*args, "-o", "json"))

    def validate_target(self) -> None:
        validate_scope(
            os.environ.get("env_x", ""), os.environ.get("GITHUB_REF", ""), self.sha
        )
        context = self.command(["kubectl", "config", "current-context"])
        if not context.endswith(":cluster/atez-dev-cluster"):
            raise CutoverRefusal("DEV_cluster_required")

    def continue_protected_release(self) -> bool:
        states = self.get("configmaps")["items"]
        previous = next(
            (item for item in states if item["metadata"]["name"] == STATE), None
        )
        if previous is None:
            return False
        data = previous["data"]
        if data["phase"] != "released":
            raise CutoverRefusal("unfinished_cutover_requires_reviewed_recovery")
        for app in APPS:
            pods = self.pods(app)
            if (
                len(pods) != 1
                or not self.ready(pods[0])
                or pods[0]["metadata"].get("deletionTimestamp")
            ):
                raise CutoverRefusal("known_compatible_release_required")
            if not any(
                item["image"] == f"{REPOSITORY}:{data['sha']}"
                for item in pods[0]["spec"]["containers"]
            ):
                raise CutoverRefusal("unrecognized_writer_image")
        self.indices = json.loads(data["indices"])
        self.scope = json.loads(data["scope"])
        self.container = data["container"]
        self.writer_nodes = set(json.loads(data["writer_nodes"]))
        return True

    def inventory(self, *, require_scope: bool = True) -> None:
        self.scope = json.loads(os.environ.get("ANNEX_DEV_ES_SCOPE_JSON", "{}"))
        if require_scope and self.scope.get("release_sha") != self.sha:
            raise CutoverRefusal("matching_reviewed_ES_scope_evidence_required")
        states = self.get("configmaps")["items"]
        previous = next(
            (item for item in states if item["metadata"]["name"] == STATE), None
        )
        if previous and previous["data"]["phase"] != "released":
            raise CutoverRefusal("unfinished_cutover_requires_reviewed_recovery")
        expected = {
            f"dev-customs-regulations-{app}-deployment" for app in (*APPS, "web")
        }
        deployments = self.get("deployments")["items"]
        if {item["metadata"]["name"] for item in deployments} != expected:
            raise CutoverRefusal("unexpected_DEV_deployments")
        for kind in (
            "statefulsets",
            "daemonsets",
            "jobs",
            "cronjobs",
            "horizontalpodautoscalers",
        ):
            if self.get(kind)["items"]:
                raise CutoverRefusal("unaccounted_DEV_controller")
        for pod in self.get("pods")["items"]:
            if pod["metadata"].get("labels", {}).get("app") not in {
                f"dev-customs-regulations-{app}" for app in (*APPS, "web")
            }:
                raise CutoverRefusal("unaccounted_DEV_pod")
        for app in APPS:
            deployment = self.get(
                "deployment", f"dev-customs-regulations-{app}-deployment"
            )
            if deployment["metadata"].get("ownerReferences"):
                raise CutoverRefusal("externally_managed_writer_controller")
            if deployment["spec"].get("replicas") != 1:
                raise CutoverRefusal("single_DEV_replica_required")
            pods = self.pods(app)
            if (
                len(pods) != 1
                or pods[0]["metadata"].get("deletionTimestamp")
                or not self.ready(pods[0])
            ):
                raise CutoverRefusal("stable_old_writer_required")
            self.writer_nodes.add(pods[0]["spec"]["nodeName"])
            node = self.get("node", pods[0]["spec"]["nodeName"])
            if not self.ready(node):
                raise CutoverRefusal("reachable_writer_node_required")

    @staticmethod
    def ready(item: dict[str, Any]) -> bool:
        return any(
            condition["type"] == "Ready" and condition["status"] == "True"
            for condition in item.get("status", {}).get("conditions", [])
        )

    def pods(self, app: str) -> list[dict[str, Any]]:
        return json.loads(
            self.kubectl(
                "get", "pods", "-l", f"app=dev-customs-regulations-{app}", "-o", "json"
            )
        )["items"]

    def create_probe(self) -> None:
        deployment = self.get(
            "deployment", "dev-customs-regulations-background-deployment"
        )
        template = deployment["spec"]["template"]
        containers = template["spec"]["containers"]
        if len(containers) != 1:
            raise CutoverRefusal("one_background_container_required")
        container = containers[0]
        self.container = container["name"]
        container["image"] = f"{REPOSITORY}:{self.sha}"
        container["command"] = ["sh", "-c", "exec sleep 7200"]
        container["args"] = []
        for key in ("livenessProbe", "readinessProbe", "startupProbe", "lifecycle"):
            container.pop(key, None)
        template["spec"]["restartPolicy"] = "Never"
        template["spec"]["activeDeadlineSeconds"] = 7200
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": PROBE,
                "namespace": NAMESPACE,
                "annotations": template["metadata"].get("annotations", {}),
                "labels": {"annex-cutover": "probe"},
            },
            "spec": template["spec"],
        }
        self.kubectl("create", "-f", "-", stdin=json.dumps(pod))
        self.kubectl("wait", "--for=condition=Ready", f"pod/{PROBE}", "--timeout=300s")

    def pod_exec(
        self, pod: str, container: str, phase: str, stdin: str | None = None
    ) -> str:
        if phase not in {"inventory", "inspect", "block", "unblock"}:
            raise ValueError("fixed_probe_phase_required")
        return self.command(
            [
                "kubectl",
                "--namespace",
                NAMESPACE,
                "exec",
                "-i",
                pod,
                "-c",
                container,
                "--",
                "sh",
                "-eu",
                "-c",
                '. /vault/secrets/config; exec python -m onyx.db.regulatory_annex_dev_cutover "$@"',
                "cutover",
                phase,
            ],
            stdin,
        )

    def inspect_indices(self) -> None:
        self.indices = json.loads(
            self.pod_exec(
                PROBE, self.container, "inspect", json.dumps({"scope": self.scope})
            )
        )

    def save(self, phase: str) -> None:
        value = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": STATE, "namespace": NAMESPACE},
            "data": {
                "phase": phase,
                "sha": self.sha,
                "indices": json.dumps(self.indices, sort_keys=True),
                "container": self.container,
                "scope": json.dumps(self.scope, sort_keys=True),
                "writer_nodes": json.dumps(sorted(self.writer_nodes)),
            },
        }
        self.kubectl("apply", "-f", "-", stdin=json.dumps(value))

    def record_started(self) -> None:
        self.save("quiescing")

    def record_blocked(self) -> None:
        self.save("blocked")

    def record_installing(self) -> None:
        self.save("installing")

    def record_released(self) -> None:
        self.save("released")

    def load_state(self) -> None:
        data = self.get("configmap", STATE)["data"]
        if data["sha"] != self.sha or data["phase"] != "installing":
            raise CutoverRefusal("matching_blocked_cutover_required")
        self.indices = json.loads(data["indices"])
        self.container = data["container"]
        self.scope = json.loads(data["scope"])
        self.writer_nodes = set(json.loads(data["writer_nodes"]))

    def stop(self, app: str) -> None:
        self.kubectl(
            "scale",
            f"deployment/dev-customs-regulations-{app}-deployment",
            "--replicas=0",
        )
        self.kubectl(
            "wait",
            "--for=delete",
            "pod",
            "-l",
            f"app=dev-customs-regulations-{app}",
            "--timeout=660s",
        )

    def stop_api(self) -> None:
        self.stop("api")

    def drain_background(self) -> None:
        pods = self.pods("background")
        if len(pods) != 1:
            raise CutoverRefusal("single_old_background_required")
        pod = pods[0]
        containers = [
            item
            for item in pod["spec"]["containers"]
            if item["image"].startswith(REPOSITORY + ":")
        ]
        if len(containers) != 1:
            raise CutoverRefusal("exact_background_container_required")
        script = Path(__file__).with_name("regulatory_annex_dev_drain.py").read_text()
        self.command(
            [
                "kubectl",
                "--namespace",
                NAMESPACE,
                "exec",
                "-i",
                pod["metadata"]["name"],
                "-c",
                containers[0]["name"],
                "--",
                "sh",
                "-eu",
                "-c",
                ". /vault/secrets/config; exec python -",
            ],
            script,
        )

    def stop_background(self) -> None:
        self.stop("background")

    def assert_no_writers(self) -> None:
        if self.get("horizontalpodautoscalers")["items"]:
            raise CutoverRefusal("writer_autoscaler_appeared")
        for node in self.writer_nodes:
            if not self.ready(self.get("node", node)):
                raise CutoverRefusal("old_writer_node_unreachable")
        for app in APPS:
            if (
                self.pods(app)
                or self.get("deployment", f"dev-customs-regulations-{app}-deployment")[
                    "spec"
                ]["replicas"]
                != 0
            ):
                raise CutoverRefusal("old_writers_remain")

    def block_indices(self) -> None:
        self.pod_exec(
            PROBE,
            self.container,
            "block",
            json.dumps({"indices": self.indices, "scope": self.scope}),
        )

    def unblock_indices(self) -> None:
        self.pod_exec(
            PROBE,
            self.container,
            "unblock",
            json.dumps({"indices": self.indices, "scope": self.scope}),
        )

    def verify_runtime(self) -> None:
        for app in APPS:
            pods = self.pods(app)
            if (
                len(pods) != 1
                or not self.ready(pods[0])
                or pods[0]["metadata"].get("deletionTimestamp")
            ):
                raise CutoverRefusal("one_ready_compatible_pod_required")
            pod = pods[0]
            containers = [
                item
                for item in pod["spec"]["containers"]
                if item["image"] == f"{REPOSITORY}:{self.sha}"
            ]
            if len(containers) != 1:
                raise CutoverRefusal("compatible_exact_SHA_required")
            if any(
                item.get("restartCount", 0)
                for item in pod["status"]["containerStatuses"]
            ):
                raise CutoverRefusal("new_pod_restarted")
            if app == "background":
                self.command(
                    [
                        "kubectl",
                        "--namespace",
                        NAMESPACE,
                        "exec",
                        pod["metadata"]["name"],
                        "-c",
                        containers[0]["name"],
                        "--",
                        "sh",
                        "-eu",
                        "-c",
                        ". /vault/secrets/config; exec python -m onyx.background.celery.regulatory_annex_readiness",
                    ]
                )

    def delete_probe(self) -> None:
        self.kubectl(
            "delete",
            "pod",
            PROBE,
            "--ignore-not-found=true",
            "--wait=true",
            "--timeout=120s",
        )

    def failure(self) -> None:
        self.validate_target()
        data = self.get("configmap", STATE)["data"]
        if data["sha"] != self.sha:
            raise CutoverRefusal("failure_state_SHA_mismatch")
        # After release only compatible images exist. Never restore old Helm revisions.
        self.stop_api()
        self.drain_background()
        self.stop_background()


def render_values(enabled: bool = False) -> None:
    import yaml

    for app in APPS:
        path = Path(
            f"devops/dev/customs-regulations/customs-regulations-{app}-values.yaml"
        )
        values = yaml.safe_load(path.read_text())
        environment = values["app"].setdefault("environment", {})
        environment["enabled"] = True
        parameters = environment.setdefault("parameters", [])
        updates = {
            "REGULATORY_ANNEX_WORKER_ENABLED": "true",
            "REGULATORY_ANNEX_ENVIRONMENT": "dev",
            "REGULATORY_ANNEX_UPDATES_ENABLED": str(enabled).lower(),
        }
        for name, value in updates.items():
            matches = [entry for entry in parameters if entry["name"] == name]
            if len(matches) > 1:
                raise CutoverRefusal("duplicate_environment_parameter")
            if matches:
                matches[0]["value"] = value
            else:
                parameters.append({"name": name, "value": value})
        path.write_text(yaml.safe_dump(values, sort_keys=False))


def inventory_only(driver: Driver) -> None:
    driver.validate_target()
    driver.inventory(require_scope=False)
    driver.create_probe()
    evidence = json.loads(driver.pod_exec(PROBE, driver.container, "inventory", "{}"))
    print(json.dumps(evidence, sort_keys=True))
    driver.delete_probe()


def require_release_runs(sha: str) -> None:
    repository = os.environ.get("GITHUB_REPOSITORY")
    if repository != "atezsoftware/customs-regulations-chatbot":
        raise CutoverRefusal("exact_repository_required")
    for workflow in (
        "customs-regulations-backend-lite-codebuild.yaml",
        "customs-regulations-web-codebuild.yaml",
    ):
        request = urllib.request.Request(
            f"https://api.github.com/repos/{repository}/actions/workflows/{workflow}/runs?head_sha={sha}&branch=develop&event=push&per_page=100",
            headers={
                "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                "Accept": "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 -- fixed HTTPS GitHub origin
            runs = json.load(response)["workflow_runs"]
        matching = [
            run
            for run in runs
            if run["head_sha"] == sha
            and run["head_branch"] == "develop"
            and run["event"] == "push"
        ]
        if (
            not matching
            or max(matching, key=lambda run: run["id"])["conclusion"] != "success"
        ):
            raise CutoverRefusal("both_exact_SHA_push_workflows_required")


def verify_frontend(driver: Driver) -> None:
    pods = driver.pods("web")
    image = f"255114580789.dkr.ecr.eu-central-1.amazonaws.com/customs-regulations-web-dev:{driver.sha}"
    if (
        len(pods) != 1
        or not driver.ready(pods[0])
        or pods[0]["metadata"].get("deletionTimestamp")
        or not any(
            container["image"] == image for container in pods[0]["spec"]["containers"]
        )
    ):
        raise CutoverRefusal("exact_SHA_web_required")
    with urllib.request.urlopen(
        "https://dev-customs-regulations.singlewindow.io/api/health", timeout=30
    ) as response:
        if response.status != 200:
            raise CutoverRefusal("frontend_API_health_required")


def acceptance(driver: Driver, phase: str) -> None:
    # Task6 supplies this fixed, reviewed provider/parser/canary module in the image.
    # Its absence fails the command; an operator cannot substitute arbitrary code.
    if phase not in {"preflight", "canary"}:
        raise ValueError("fixed_acceptance_phase_required")
    if not re.fullmatch(r"[0-9a-f]{40}", driver.sha):
        raise ValueError("exact_acceptance_release_sha_required")
    pod = driver.pods("background")[0]
    container = next(
        item
        for item in pod["spec"]["containers"]
        if item["image"] == f"{REPOSITORY}:{driver.sha}"
    )
    driver.command(
        [
            "kubectl",
            "--namespace",
            NAMESPACE,
            "exec",
            pod["metadata"]["name"],
            "-c",
            container["name"],
            "--",
            "env",
            "ANNEX_ACCEPTANCE_RELEASE_SHA=" + driver.sha,
            "sh",
            "-eu",
            "-c",
            '. /vault/secrets/config; exec python -m onyx.regulatory.amendments.annexes.dev_acceptance "$@"',
            "acceptance",
            phase,
        ],
        acceptance_phase=phase,
    )


def deploy_same_image(driver: Driver, enabled: bool) -> None:
    render_values(enabled)
    for app in APPS:
        args = [
            "helm",
            "upgrade",
            "--install",
            f"customs-regulations-{app}-dev",
            "-f",
            f"devops/dev/customs-regulations/customs-regulations-{app}-values.yaml",
            "--namespace",
            NAMESPACE,
            "--server-side=false",
            "--set-string",
            f"image.tag={driver.sha}",
            "--timeout",
            "600s",
        ]
        if app == "background":
            args += ["--set", "app.mem_limits=4Gi"]
        driver.command([*args, "./devops"])
        driver.kubectl(
            "rollout",
            "status",
            f"deployment/dev-customs-regulations-{app}-deployment",
            "--timeout=600s",
        )


def verify_or_activate(driver: Driver, activate: bool) -> None:
    driver.validate_target()
    state = driver.get("configmap", STATE)["data"]
    if state["sha"] != driver.sha or state["phase"] != "released":
        raise CutoverRefusal("successful_matching_cutover_required")
    require_release_runs(driver.sha)
    driver.verify_runtime()
    verify_frontend(driver)
    acceptance(driver, "preflight")
    if activate:
        try:
            deploy_same_image(driver, True)
            driver.verify_runtime()
            verify_frontend(driver)
            acceptance(driver, "canary")
        except Exception:
            # Same reviewed binary; disable creation without undoing protected data.
            deploy_same_image(driver, False)
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=(
            "prepare",
            "release",
            "failure",
            "values",
            "verify",
            "annex-inventory",
            "annex-verify",
            "annex-activate",
        ),
    )
    args = parser.parse_args()
    sha = os.environ.get("IMAGE_TAG", "")
    validate_scope(os.environ.get("env_x", ""), os.environ.get("GITHUB_REF", ""), sha)
    if sha != os.environ.get("GITHUB_SHA"):
        raise CutoverRefusal("image_must_match_checked_out_workflow_SHA")
    driver = Driver(sha)
    if args.phase == "prepare":
        prepare(driver)
    elif args.phase == "release":
        release(driver)
    elif args.phase == "failure":
        driver.failure()
    elif args.phase == "values":
        render_values()
    elif args.phase == "annex-inventory":
        inventory_only(driver)
    elif args.phase in {"annex-verify", "annex-activate"}:
        verify_or_activate(driver, args.phase == "annex-activate")
    else:
        driver.validate_target()
        driver.verify_runtime()
    print(f"DEV_CUTOVER_{args.phase.upper()}_OK")


if __name__ == "__main__":
    try:
        main()
    except CutoverRefusal as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
    except Exception:
        print(
            "DEV_CUTOVER_REFUSED: preserve state and compatible binaries; no Helm rollback",
            file=sys.stderr,
        )
        sys.exit(1)
