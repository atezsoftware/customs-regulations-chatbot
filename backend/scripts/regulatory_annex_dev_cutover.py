"""Bounded runner entrypoint for the existing DEV backend-lite deployment."""

import argparse
import inspect
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from onyx.db.regulatory_annex_acceptance_diagnostic import load_source861_canary

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
    timing_keys = {
        f"stage_{stage}_{metric}"
        for stage in (
            "baseline",
            "source_review",
            "approval",
            "historical_chat",
            "current_chat",
            "markdown",
        )
        for metric in ("elapsed_ms", "remaining_start_ms", "remaining_end_ms")
    }
    progress_counts = {
        "markdown_job_" + key
        for key in (
            "attempt_count",
            "total_items",
            "completed_items",
            "context_ready_items",
            "embedded_items",
            "failed_items",
        )
    }
    progress_bools = {
        "markdown_upload_accepted",
        "markdown_index_post_attempted",
        "markdown_index_post_accepted",
        "markdown_job_present",
        "progress_save_failed",
    }
    progress_enums = {
        "markdown_last_status": {
            "UNKNOWN",
            "PROCESSING",
            "INDEXING",
            "CHUNKED",
            "COMPLETED",
            "SKIPPED",
            "FAILED",
            "CANCELED",
            "DELETING",
        },
        "markdown_job_status": {
            "UNKNOWN",
            "QUEUED",
            "RUNNING",
            "RETRY_WAIT",
            "SUCCEEDED",
            "FAILED",
            "CANCELLING",
            "CANCELLED",
        },
        "markdown_job_stage": {
            "UNKNOWN",
            "PREPARING",
            "CONTEXT_SUBMIT",
            "CONTEXT_WAIT",
            "CONTEXT_APPLY",
            "EMBEDDING",
            "INDEX_WRITE",
            "VERIFY",
            "PUBLISH",
        },
    }
    keys = {
        "phase",
        "status",
        "release_sha_metadata",
        "failure_stage",
        "exception_type",
        "configuration",
        "native",
        "calibration",
        "pdf_vision_probe",
        "probe_stage",
        "native_value_absent",
        "image_evidence",
        "grounding_verified",
        "page_count",
        "transcript_sha256",
        "draft_sha256",
        "receipt_sha256",
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
        "failure_detail",
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
        "persona_id",
        "persona_cleanup_complete",
        "persona_cleanup_failure",
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
        "source_package_status",
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
        "ordinary_markdown_upload_index_chat",
        "acceptance_passed",
        "retained_tombstones",
        "cleanup_live_projections",
        "retained_source_scope",
        "worker_failure",
        "cleanup_failure",
        "chat_cleanup_failure",
        "token_cleanup_failure",
        "chat_2026-09-09",
        "chat_2026-09-10",
    }
    keys.update(
        timing_keys
        | progress_counts
        | progress_bools
        | progress_enums.keys()
        | {"markdown_poll_count"}
    )

    def sanitize(value: Any, parent: str = "", depth: int = 0) -> Any:
        if depth > 8:
            raise CutoverRefusal("acceptance_report_depth_exceeded")
        if parent in timing_keys | progress_counts | {"markdown_poll_count"}:
            maximum = (
                86400000
                if parent in timing_keys
                else 10000
                if parent == "markdown_poll_count"
                else 2147483647
            )
            if type(value) is not int or not 0 <= value <= maximum:
                raise CutoverRefusal("acceptance_progress_value_refused")
        if parent in progress_bools and type(value) is not bool:
            raise CutoverRefusal("acceptance_progress_value_refused")
        if parent in progress_enums and (
            not isinstance(value, str) or value not in progress_enums[parent]
        ):
            raise CutoverRefusal("acceptance_progress_value_refused")
        if parent == "source_package_status" and value not in ("failed", "blocked"):
            raise CutoverRefusal("acceptance_source_package_status_refused")
        if parent == "failure_stage" and (
            not isinstance(value, str)
            or value
            not in {
                "scope",
                "startup",
                "configuration",
                "native",
                "calibration",
                "pdf_vision",
                "canary",
            }
        ):
            raise CutoverRefusal("acceptance_failure_stage_refused")
        if parent == "exception_type" and (
            not isinstance(value, str)
            or value
            not in {
                "UnicodeDecodeError",
                "ValueError",
                "ValidationError",
                "RuntimeError",
                "TypeError",
                "TimeoutError",
                "OperationalError",
                "ImportError",
                "ModuleNotFoundError",
                "IsolatedProcessTimeout",
                "IsolatedProcessCrashed",
                "AssertionError",
                "KeyError",
                "Exception",
            }
        ):
            raise CutoverRefusal("acceptance_exception_type_refused")
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
        operation = (
            next(
                (part for part in args[1:] if part in {"exec", "rollout", "get"}),
                "other",
            )
            if args[0] == "kubectl"
            else "other"
        )
        try:
            result = subprocess.run(
                args, input=stdin, text=True, capture_output=True, timeout=timeout
            )
        except (subprocess.TimeoutExpired, OSError):
            print(f"command_transport_failed:{args[0]}:{operation}", flush=True)
            raise CutoverRefusal(
                f"command_transport_failed:{args[0]}:{operation}"
            ) from None
        if acceptance_phase is not None:
            if acceptance_phase not in {"preflight", "canary"}:
                raise CutoverRefusal("fixed_acceptance_phase_required")
            emit_acceptance_report(result.stdout, acceptance_phase, self.sha)
        if result.returncode:
            # kubectl/provider errors can contain credentials or source text.
            print(f"command_failed:{args[0]}:{operation}", flush=True)
            raise CutoverRefusal(f"command_failed:{args[0]}:{operation}")
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
                "get",
                "pods",
                "-l",
                f"app=dev-customs-regulations-{app}",
                "-o",
                "json",
                "--request-timeout=10s",
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

    def verify_runtime(self, *, readiness: bool = True) -> None:
        for app in APPS:
            self.verify_app(app, readiness=readiness)

    def verify_app(self, app: str, *, readiness: bool = True) -> None:
        if app not in APPS:
            raise CutoverRefusal("fixed_runtime_app_required")
        worker_required = app == "background" and readiness
        deadline = time.monotonic() + (600 if worker_required else 180)
        while True:
            pods = self.pods(app)
            wrong_sha = sum(
                sum(
                    container["image"] == f"{REPOSITORY}:{self.sha}"
                    for container in pod["spec"]["containers"]
                )
                != 1
                for pod in pods
            )
            if wrong_sha:
                raise CutoverRefusal("compatible_exact_SHA_required")
            if any(
                item.get("restartCount", 0)
                for pod in pods
                for item in pod["status"].get("containerStatuses", [])
            ):
                raise CutoverRefusal("new_pod_restarted")
            if (
                len(pods) == 1
                and self.ready(pods[0])
                and not pods[0]["metadata"].get("deletionTimestamp")
            ):
                # A Ready survivor must expose the same restart evidence as before.
                if not pods[0]["status"].get("containerStatuses"):
                    raise CutoverRefusal("runtime_container_status_required")
                if not worker_required:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CutoverRefusal("worker_readiness_timeout")
                if self.worker_readiness_probe(
                    pods[0], timeout=min(60, max(1, int(remaining)))
                ):
                    break
                if time.monotonic() >= deadline:
                    raise CutoverRefusal("worker_readiness_timeout")
                time.sleep(5)
                continue
            if time.monotonic() >= deadline:
                print(
                    json.dumps(
                        {
                            "stage": "runtime_settle_timeout",
                            "app": app,
                            "pods": len(pods),
                            "ready": sum(self.ready(pod) for pod in pods),
                            "deleting": sum(
                                bool(pod["metadata"].get("deletionTimestamp"))
                                for pod in pods
                            ),
                            "wrong_sha": wrong_sha,
                        },
                        sort_keys=True,
                    )
                )
                raise CutoverRefusal("one_ready_compatible_pod_required")
            time.sleep(2)

    def worker_readiness_probe(self, pod: dict[str, Any], *, timeout: int) -> bool:
        container = next(
            item
            for item in pod["spec"]["containers"]
            if item["image"] == f"{REPOSITORY}:{self.sha}"
        )
        args = [
            "kubectl",
            "--namespace",
            NAMESPACE,
            "exec",
            pod["metadata"]["name"],
            "-c",
            container["name"],
            "--",
            "sh",
            "-eu",
            "-c",
            ". /vault/secrets/config; exec python -m onyx.background.celery.regulatory_annex_readiness",
        ]
        try:
            result = subprocess.run(
                args, text=True, capture_output=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            print("NOT_READY annex_worker ProbeTimeout", flush=True)
            return False
        except OSError:
            print("NOT_READY annex_worker ExecTransportFailure", flush=True)
            raise CutoverRefusal("command_failed:kubectl:exec") from None
        reports = [
            line
            for line in result.stdout.splitlines()
            if line.startswith(("READY annex_worker", "NOT_READY annex_worker"))
        ]
        ready = "READY annex_worker scoped_queues registered_handlers concurrency_one"
        if len(reports) == 1:
            report = reports[0]
            if result.returncode == 0 and report == ready:
                print(ready, flush=True)
                return True
            if result.returncode == 1 and re.fullmatch(
                r"NOT_READY annex_worker [A-Za-z_][A-Za-z0-9_]{0,99}", report
            ):
                print(report, flush=True)
                return False
        print("NOT_READY annex_worker InvalidFixedReport", flush=True)
        raise CutoverRefusal("worker_readiness_report_required:kubectl:exec")

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


RUNNER_ONLY_PATHS = frozenset(
    {
        "backend/scripts/regulatory_annex_dev_cutover.py",
        ".github/workflows/customs-regulations-backend-lite-codebuild.yaml",
        "backend/tests/unit/scripts/test_regulatory_annex_dev_cutover.py",
    }
)


def require_runner_compatibility(runtime_sha: str, runner_sha: str) -> None:
    """Accept only a complete, ancestor-based GitHub comparison of runner files."""
    repository = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GH_TOKEN")
    if repository != "atezsoftware/customs-regulations-chatbot" or not token:
        raise CutoverRefusal("authenticated_exact_repository_required")
    if not all(re.fullmatch("[0-9a-f]{40}", sha) for sha in (runtime_sha, runner_sha)):
        raise CutoverRefusal("exact_comparison_SHAs_required")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/compare/{runtime_sha}...{runner_sha}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 -- fixed HTTPS GitHub origin
            if response.status != 200 or response.headers.get("Link"):
                raise ValueError
            raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError
            comparison = json.loads(raw)
        commits = comparison["commits"]
        files = comparison["files"]
        total = comparison["total_commits"]
        # GitHub's unpaginated comparison caps commits at 250 and files at 300.
        if (
            comparison["status"] != "ahead"
            or comparison["behind_by"] != 0
            or comparison["ahead_by"] != total
            or comparison["base_commit"]["sha"] != runtime_sha
            or comparison["merge_base_commit"]["sha"] != runtime_sha
            or comparison.get("truncated", False) is not False
            or type(total) is not int
            or not 0 < total <= 250
            or not isinstance(commits, list)
            or len(commits) != total
            or commits[-1]["sha"] != runner_sha
            or any(
                not re.fullmatch("[0-9a-f]{40}", commit["sha"]) for commit in commits
            )
            or len({commit["sha"] for commit in commits}) != total
            or not isinstance(files, list)
            or not 0 < len(files) < 300
            or any(
                file["filename"] not in RUNNER_ONLY_PATHS
                or file["status"] != "modified"
                or "previous_filename" in file
                for file in files
            )
            or len({file["filename"] for file in files}) != len(files)
        ):
            raise ValueError
    except Exception:
        raise CutoverRefusal("runner_only_comparison_required") from None
    print(
        json.dumps(
            {
                "runner_sha": runner_sha,
                "runtime_sha": runtime_sha,
                "runner_only_comparison": "verified",
            },
            sort_keys=True,
        )
    )


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
    request = urllib.request.Request(
        "https://dev-customs-regulations.singlewindow.io/api/health",
        headers={"User-Agent": "Onyx-DEV-Release/1.0", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 -- fixed DEV HTTPS origin
            if response.status != 200:
                raise CutoverRefusal("frontend_API_health_required")
    except urllib.error.HTTPError as error:
        raise CutoverRefusal(f"frontend_API_health_HTTP_{error.code}") from None


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


DIAGNOSTIC_PROGRAM = r"""
import contextlib
import io
import json
import os
import platform
import re


def scope():
    from onyx.regulatory.amendments.annexes.dev_acceptance import validate_scope
    validate_scope(database=os.environ.get("POSTGRES_DB", ""),
                   environment=os.environ.get("REGULATORY_ANNEX_ENVIRONMENT", ""),
                   machine=platform.machine())


def configuration():
    scope()
    from onyx.db.regulatory_annex_acceptance import verify_dev_configuration
    value = verify_dev_configuration()
    if value.get("database") != "customs-regulations-dev":
        raise ValueError()
    return {"database": value["database"], "indices": value["indices"]}


def native_parser():
    scope()
    from onyx.regulatory.amendments.annexes.dev_acceptance import native_parser_probe
    native_parser_probe()


def run(stage, function):
    result = {"stage": stage, "exception_class": None, "frames": []}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            value = function()
            if stage == "configuration":
                result["configuration"] = value
        except Exception as error:
            result["exception_class"] = type(error).__name__
            trace = error.__traceback__
            while trace is not None:
                filename = trace.tb_frame.f_code.co_filename.replace("\\", "/")
                match = re.search(r"(?:^|/)((?:onyx|shared_configs|ee/onyx)/[A-Za-z0-9_./-]+\.py)$", filename)
                if match and ".." not in match.group(1).split("/"):
                    result["frames"].append({"filename": match.group(1), "function": trace.tb_frame.f_code.co_name, "line": trace.tb_lineno})
                trace = trace.tb_next
    return result


print(json.dumps({"stages": [run("configuration", configuration), run("native_parser", native_parser)]}, sort_keys=True))
"""


BATCH44_RUNTIME = "4a38658efa5014dfa2b039887bc7db0e12e8153e"
BATCH44_GROUP = "/aws/containerinsights/atez-dev-cluster/application"
BATCH44_START, BATCH44_END = 1789123620000, 1789123800000
BATCH44_FILTER = '{ $.kubernetes.namespace_name = "customs-regulations-dev" && $.kubernetes.pod_name = %^dev-customs-regulations-background-% }'


def cloudwatch_metadata(args: list[str]) -> dict[str, Any]:
    """Fixed diagnostic callers only; never expose AWS output or error messages."""
    try:
        response = subprocess.run(
            [
                "aws",
                "logs",
                *args,
                "--region",
                "eu-central-1",
                "--output",
                "json",
                "--no-cli-pager",
                "--no-paginate",
                "--cli-connect-timeout",
                "5",
                "--cli-read-timeout",
                "15",
            ],
            text=True,
            capture_output=True,
            timeout=20,
        )
        if response.returncode:
            denied = (
                "AccessDenied" in response.stderr
                or "UnauthorizedOperation" in response.stderr
            )
            raise CutoverRefusal("access_denied" if denied else "aws_query_failed")
        if len(response.stdout) > 2_000_000:
            raise CutoverRefusal("response_too_large")
        value = json.loads(response.stdout)
        if not isinstance(value, dict):
            raise CutoverRefusal("invalid_response")
        return value
    except (OSError, subprocess.TimeoutExpired):
        raise CutoverRefusal("aws_transport_unavailable") from None
    except (ValueError, TypeError):
        raise CutoverRefusal("invalid_response") from None


def diagnose_logging_metadata(driver: Driver) -> None:
    """Bounded collector/destination inventory; no application log reads."""
    report: dict[str, Any] = {
        "stage": "logging_destination_metadata",
        "collectors": [],
        "cloudwatch": [],
    }
    fluentd: dict[str, Any] | None = None
    for namespace in ("amazon-cloudwatch", "kube-system", "logging", "monitoring"):
        result: dict[str, Any] = {
            "namespace": namespace,
            "status": "available",
            "daemonsets": [],
        }
        try:
            data = json.loads(
                driver.command(
                    [
                        "kubectl",
                        "get",
                        "daemonsets",
                        "--namespace",
                        namespace,
                        "-o",
                        "json",
                        "--request-timeout=10s",
                    ],
                    timeout=15,
                )
            )
            for item in data["items"]:
                name = item["metadata"]["name"]
                if namespace == "logging" and name == "fluentd":
                    fluentd = item["spec"]["template"]["spec"]
                images = [
                    container["image"]
                    for container in item["spec"]["template"]["spec"]["containers"]
                ]
                if not re.fullmatch(r"[a-z0-9.-]{1,253}", name) or not all(
                    re.fullmatch(r"[A-Za-z0-9._/@:-]{1,400}", image) for image in images
                ):
                    raise ValueError
                if re.search(
                    r"fluent|cloudwatch|filebeat|promtail|vector|opentelemetry|otel",
                    " ".join([name, *images]),
                    re.IGNORECASE,
                ):
                    result["daemonsets"].append({"name": name, "images": images})
        except (CutoverRefusal, ValueError, TypeError, KeyError):
            result.update(status="unavailable", daemonsets=[])
        report["collectors"].append(result)
    if fluentd is not None:
        destination: dict[str, Any] = {
            "status": "available",
            "configmaps": [],
            "destinations": [],
            "services": [],
        }
        fields = {
            "host": r"[A-Za-z0-9.-]{1,253}",
            "port": r"[0-9]{1,5}",
            "scheme": r"https?",
            "index_name": r"[A-Za-z0-9_.-]{1,253}",
            "logstash_prefix": r"[A-Za-z0-9_.-]{1,253}",
        }
        try:
            references = set()
            values: list[tuple[str, str]] = []
            for volume in fluentd.get("volumes", []):
                if "configMap" in volume:
                    references.add(volume["configMap"]["name"])
                for source in volume.get("projected", {}).get("sources", []):
                    if "configMap" in source:
                        references.add(source["configMap"]["name"])
            for container in fluentd["containers"]:
                for source in container.get("envFrom", []):
                    if "configMapRef" in source:
                        references.add(source["configMapRef"]["name"])
                for env in container.get("env", []):
                    if "configMapKeyRef" in env.get("valueFrom", {}):
                        references.add(env["valueFrom"]["configMapKeyRef"]["name"])
                    key = (
                        env.get("name", "")
                        .removeprefix("FLUENT_ELASTICSEARCH_")
                        .lower()
                    )
                    if key in fields and isinstance(env.get("value"), str):
                        values.append((key, env["value"]))
            if len(references) > 8 or not all(
                isinstance(name, str)
                and re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,252}", name)
                and ".." not in name
                for name in references
            ):
                raise CutoverRefusal("invalid_references")
            for name in sorted(references):
                config = json.loads(
                    driver.command(
                        [
                            "kubectl",
                            "get",
                            "configmap",
                            name,
                            "--namespace",
                            "logging",
                            "-o",
                            "json",
                            "--request-timeout=10s",
                        ],
                        timeout=15,
                    )
                )
                if config.get("metadata", {}).get("name") != name:
                    raise CutoverRefusal("referenced_config_unavailable")
                destination["configmaps"].append(name)
                for key, value in config["data"].items():
                    if not isinstance(value, str):
                        raise ValueError
                    field = key.removeprefix("FLUENT_ELASTICSEARCH_").lower()
                    if field in fields:
                        values.append((field, value))
                    if re.search(r"(?m)^\s*@type\s+elasticsearch\s*$", value):
                        values.extend(
                            re.findall(
                                r"(?m)^\s*(host|port|scheme|index_name|logstash_prefix)\s+([^\r\n]+)$",
                                value,
                            )
                        )
            for field, value in values:
                value = value.strip().strip("\"'")
                if re.fullmatch(fields[field], value) and (
                    field != "port" or 0 < int(value) <= 65535
                ):
                    destination["destinations"].append({"field": field, "value": value})
            services = json.loads(
                driver.command(
                    [
                        "kubectl",
                        "get",
                        "services",
                        "--namespace",
                        "logging",
                        "-o",
                        "json",
                        "--request-timeout=10s",
                    ],
                    timeout=15,
                )
            )
            for service in services["items"]:
                name = service["metadata"]["name"]
                if not re.fullmatch(r"[a-z0-9.-]{1,253}", name) or not re.search(
                    r"elastic|kibana", name
                ):
                    continue
                address = service["spec"].get("clusterIP", "")
                ports = [port["port"] for port in service["spec"].get("ports", [])]
                if not re.fullmatch(r"[0-9a-fA-F:.]{1,45}|None", address) or not all(
                    type(port) is int and 0 < port <= 65535 for port in ports
                ):
                    continue
                destination["services"].append(
                    {"name": name, "cluster_ip": address, "ports": ports}
                )
        except CutoverRefusal as error:
            destination["status"] = (
                str(error)
                if str(error) in {"invalid_references", "referenced_config_unavailable"}
                else "unavailable"
            )
        except (ValueError, TypeError, KeyError, AttributeError):
            destination["status"] = "invalid_metadata"
        report["fluentd_destination"] = destination
    try:
        config = json.loads(
            driver.command(
                [
                    "kubectl",
                    "get",
                    "configmap",
                    "aws-logging",
                    "--namespace",
                    "aws-observability",
                    "-o",
                    "json",
                    "--request-timeout=10s",
                ],
                timeout=15,
            )
        )
        destinations = []
        for value in config["data"].values():
            if not isinstance(value, str):
                raise ValueError
            for key, destination in re.findall(
                r"(?im)^\s*(log_group_name|region)\s+([A-Za-z0-9._/#-]{1,512})\s*$",
                value,
            ):
                destinations.append({"field": key.lower(), "value": destination})
        report["fargate"] = {"status": "available", "destinations": destinations}
    except (CutoverRefusal, ValueError, TypeError, KeyError, AttributeError):
        report["fargate"] = {"status": "unavailable", "destinations": []}
    for pattern in ("atez-dev", "customs-regulations-dev"):
        entry: dict[str, Any] = {
            "pattern": pattern,
            "status": "available",
            "groups": [],
            "complete": False,
        }
        try:
            result = cloudwatch_metadata(
                [
                    "describe-log-groups",
                    "--log-group-name-pattern",
                    pattern,
                    "--limit",
                    "50",
                ]
            )
            names = [group["logGroupName"] for group in result["logGroups"]]
            if not all(
                isinstance(name, str)
                and pattern in name
                and re.fullmatch(r"[A-Za-z0-9._/#-]{1,512}", name)
                for name in names
            ):
                raise ValueError
            entry.update(groups=names, complete=not bool(result.get("nextToken")))
        except CutoverRefusal as error:
            entry["status"] = str(error)
        except (ValueError, TypeError, KeyError):
            entry["status"] = "invalid_response"
        report["cloudwatch"].append(entry)
    print(json.dumps(report, sort_keys=True), flush=True)


def sanitize_batch44_trace(streams: dict[str, list[str]]) -> dict[str, Any]:
    traces = []
    for lines in streams.values():
        for position, line in enumerate(lines):
            if not re.search(r"Amendment batch 44 failed(?:\x1b\[[0-9;]*m)*$", line):
                continue
            frames: list[dict[str, Any]] = []
            started = False
            for line in lines[position + 1 : position + 101]:
                line = re.sub(r"\x1b\[[0-9;]*m", "", line)
                if line == "Traceback (most recent call last):":
                    started = True
                    continue
                if not started:
                    break
                frame = re.fullmatch(
                    r'  File "[^"\n]*?((?:onyx|shared_configs|ee/onyx)/[A-Za-z0-9_./-]+\.py)", line ([1-9][0-9]*), in ([A-Za-z0-9_<>]+)',
                    line,
                )
                if frame and ".." not in frame.group(1).split("/"):
                    frames.append(
                        {
                            "filename": frame.group(1),
                            "line": int(frame.group(2)),
                            "function": frame.group(3),
                        }
                    )
                error = re.match(r"^([A-Za-z_][A-Za-z0-9_.]{0,99}):(?: |$)", line)
                if error:
                    if frames:
                        traces.append(
                            {
                                "frames": frames,
                                "exception_class": error.group(1).rsplit(".", 1)[-1],
                            }
                        )
                    break
                if line and not line.startswith(" "):
                    break
    if len(traces) != 1:
        raise CutoverRefusal(
            "batch_trace_unavailable" if not traces else "batch_trace_ambiguous"
        )
    return {**traces[0], "status": "trace_found"}


def diagnose_batch44_logs(runtime_sha: str) -> str:
    """Read only the known failed release's fixed DEV namespace/time window."""
    report: dict[str, Any] = {
        "stage": "batch44_cloudwatch",
        "status": "not_applicable",
        "frames": [],
        "exception_class": None,
    }
    if runtime_sha != BATCH44_RUNTIME:
        print(json.dumps(report, sort_keys=True), flush=True)
        return "not_applicable"
    try:
        groups = cloudwatch_metadata(
            [
                "describe-log-groups",
                "--log-group-name-prefix",
                "/aws/containerinsights/atez-dev-cluster/",
                "--limit",
                "50",
            ]
        )
        if groups.get("nextToken"):
            raise CutoverRefusal("group_inventory_incomplete")
        if not isinstance(groups.get("logGroups"), list):
            raise CutoverRefusal("invalid_response")
        if not any(
            group.get("logGroupName") == BATCH44_GROUP for group in groups["logGroups"]
        ):
            raise CutoverRefusal("application_group_absent")
        report["application_group_found"] = True
        deadline = time.monotonic() + 120
        token: str | None = None
        streams: dict[str, list[str]] = {}
        seen: set[str] = set()
        for _ in range(6):
            if time.monotonic() >= deadline:
                raise CutoverRefusal("pagination_incomplete")
            args = [
                "filter-log-events",
                "--log-group-name",
                BATCH44_GROUP,
                "--start-time",
                str(BATCH44_START),
                "--end-time",
                str(BATCH44_END),
                "--filter-pattern",
                BATCH44_FILTER,
                "--limit",
                "10000",
            ]
            if token:
                args += ["--next-token", token]
            page = cloudwatch_metadata(args)
            if not isinstance(page.get("events"), list):
                raise CutoverRefusal("invalid_response")
            for event in page["events"]:
                if (
                    not isinstance(event, dict)
                    or type(event.get("timestamp")) is not int
                    or not BATCH44_START <= event["timestamp"] < BATCH44_END
                ):
                    raise CutoverRefusal("invalid_response")
                value = json.loads(event["message"])
                kubernetes = value.get("kubernetes", {})
                if kubernetes.get("namespace_name") != NAMESPACE or not re.fullmatch(
                    r"dev-customs-regulations-background-[a-z0-9-]+",
                    kubernetes.get("pod_name", ""),
                ):
                    raise CutoverRefusal("namespace_scope_mismatch")
                if not isinstance(value.get("log"), str) or not isinstance(
                    event.get("logStreamName"), str
                ):
                    raise CutoverRefusal("invalid_response")
                streams.setdefault(event["logStreamName"], []).extend(
                    value["log"].splitlines()
                )
            token = page.get("nextToken")
            if not token:
                break
            if not isinstance(token, str) or token in seen:
                raise CutoverRefusal("pagination_incomplete")
            seen.add(token)
        else:
            raise CutoverRefusal("pagination_incomplete")
        report.update(sanitize_batch44_trace(streams))
    except CutoverRefusal as error:
        report["status"] = str(error)
    except (ValueError, TypeError, KeyError, AttributeError):
        report["status"] = "invalid_response"
    print(json.dumps(report, sort_keys=True), flush=True)
    return str(report["status"])


BATCH44_ES_PROGRAM = r"""
import base64
import json
import os
import re
import ssl
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any

class CutoverRefusal(RuntimeError):
    pass

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

def main():
    report = {"stage": "batch44_elasticsearch", "status": "unavailable", "frames": [], "exception_class": None}
    try:
        filters = [
            {"bool": {"minimum_should_match": 1, "should": [{"term": {field: "customs-regulations-dev"}} for field in ("kubernetes.namespace_name.keyword", "kubernetes.namespace_name")]}},
            {"bool": {"minimum_should_match": 1, "should": [{"regexp": {field: "dev-customs-regulations-background-.*"}} for field in ("kubernetes.pod_name.keyword", "kubernetes.pod_name")]}},
            {"range": {"@timestamp": {"gte": "2026-09-11T10:47:00Z", "lt": "2026-09-11T10:50:00Z"}}},
        ]
        body = {"size": 2000, "track_total_hits": True, "timeout": "15s", "query": {"bool": {"filter": filters}}, "sort": [{"@timestamp": {"order": "asc", "unmapped_type": "date"}}], "_source": ["@timestamp", "kubernetes.namespace_name", "kubernetes.pod_name", "log", "message"]}
        headers = {"Content-Type": "application/json"}
        matching = os.environ.get("ELASTICSEARCH_HOST") == "elastic.dev.singlewindow.io" and os.environ.get("ELASTICSEARCH_REST_API_PORT", "9200") == "9200" and os.environ.get("ELASTICSEARCH_USE_SSL", "false").lower() == "true"
        if matching:
            user, password = os.environ.get("ELASTICSEARCH_ADMIN_USERNAME"), os.environ.get("ELASTICSEARCH_ADMIN_PASSWORD")
            if user and password:
                headers["Authorization"] = "Basic " + base64.b64encode((user + ":" + password).encode()).decode()
        context = ssl.create_default_context(cafile=os.environ.get("ELASTICSEARCH_CA_CERTS") if matching else None)
        opener = urllib.request.build_opener(NoRedirect(), urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context))
        request = urllib.request.Request("https://elastic.dev.singlewindow.io:9200/_search?filter_path=timed_out,_shards.total,_shards.successful,_shards.failed,hits.total,hits.hits._source", data=json.dumps(body).encode(), headers=headers, method="POST")
        with opener.open(request, timeout=20) as response:
            if response.status != 200:
                raise CutoverRefusal("http_failure")
            raw = response.read(4_000_001)
        if len(raw) > 4_000_000:
            raise CutoverRefusal("response_too_large")
        data = json.loads(raw)
        shards, total, hits = data["_shards"], data["hits"]["total"], data["hits"]["hits"]
        if data.get("timed_out") is not False or type(shards.get("total")) is not int or shards["total"] < 1 or shards.get("failed") != 0 or shards.get("successful") != shards["total"] or total.get("relation") != "eq" or type(total.get("value")) is not int or not isinstance(hits, list) or total["value"] != len(hits) or len(hits) > 2000:
            raise CutoverRefusal("search_incomplete")
        streams = {}
        previous = None
        for hit in hits:
            source = hit["_source"]
            metadata = source["kubernetes"]
            pod = metadata["pod_name"]
            when = datetime.fromisoformat(source["@timestamp"].replace("Z", "+00:00"))
            if when.tzinfo is None or not 1789123620 <= when.timestamp() < 1789123800 or metadata.get("namespace_name") != "customs-regulations-dev" or not re.fullmatch(r"dev-customs-regulations-background-[a-z0-9-]+", pod):
                raise CutoverRefusal("source_scope_mismatch")
            if previous is not None and when < previous:
                raise CutoverRefusal("search_incomplete")
            previous = when
            text = source.get("log", source.get("message"))
            if not isinstance(text, str):
                raise CutoverRefusal("invalid_response")
            streams.setdefault(pod, []).extend(text.splitlines())
        report.update(sanitize_batch44_trace(streams))
    except urllib.error.HTTPError as error:
        report["status"] = "http_" + str(error.code) if error.code in (301, 302, 303, 307, 308, 400, 401, 403, 404, 429, 500, 502, 503, 504) else "http_failure"
    except CutoverRefusal as error:
        report["status"] = str(error)
    except (urllib.error.URLError, OSError, TimeoutError):
        report["status"] = "transport_unavailable"
    except (ValueError, TypeError, KeyError, AttributeError):
        report["status"] = "invalid_response"
    print(json.dumps(report, sort_keys=True))
"""


def diagnose_batch44_elasticsearch(driver: Driver, pod: str, container: str) -> None:
    if driver.sha != BATCH44_RUNTIME:
        return
    program = (
        BATCH44_ES_PROGRAM
        + "\n"
        + inspect.getsource(sanitize_batch44_trace)
        + "\nmain()\n"
    )
    output = driver.command(
        [
            "kubectl",
            "--namespace",
            NAMESPACE,
            "exec",
            pod,
            "-c",
            container,
            "--",
            "sh",
            "-eu",
            "-c",
            '. /vault/secrets/config; exec python -c "$1"',
            "batch44-elasticsearch",
            program,
        ],
        timeout=45,
    )
    if len(output.encode()) > 64000:
        raise CutoverRefusal("fixed_diagnostic_report_required")
    report = json.loads(output)
    allowed_statuses = {
        "trace_found",
        "unavailable",
        "http_failure",
        "response_too_large",
        "search_incomplete",
        "source_scope_mismatch",
        "invalid_response",
        "transport_unavailable",
        "batch_trace_unavailable",
        "batch_trace_ambiguous",
        *(
            "http_" + str(code)
            for code in (
                301,
                302,
                303,
                307,
                308,
                400,
                401,
                403,
                404,
                429,
                500,
                502,
                503,
                504,
            )
        ),
    }
    if (
        not isinstance(report, dict)
        or set(report) != {"stage", "status", "frames", "exception_class"}
        or report["stage"] != "batch44_elasticsearch"
        or report["status"] not in allowed_statuses
        or not isinstance(report["frames"], list)
    ):
        raise CutoverRefusal("fixed_diagnostic_report_required")
    if report["exception_class"] is not None and not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]{0,99}", report["exception_class"]
    ):
        raise CutoverRefusal("fixed_diagnostic_report_required")
    for frame in report["frames"]:
        if (
            not isinstance(frame, dict)
            or set(frame) != {"filename", "function", "line"}
            or not re.fullmatch(
                r"(?:onyx|shared_configs|ee/onyx)/[A-Za-z0-9_./-]+\.py",
                frame["filename"],
            )
            or ".." in frame["filename"].split("/")
            or not re.fullmatch(r"[A-Za-z0-9_<>]+", frame["function"])
            or type(frame["line"]) is not int
            or frame["line"] < 1
        ):
            raise CutoverRefusal("fixed_diagnostic_report_required")
    print(json.dumps(report, sort_keys=True), flush=True)


BATCH47_RUNTIME = "e156b5d34ac4da03013914ec8581b879368c0050"


def comparison_diagnostic_summary(
    content: str, error: Any, finish_reason: Any
) -> dict[str, Any]:
    """Keep schema shape only; response values and arbitrary field names stay private."""
    from typing import get_args

    from pydantic_core import ErrorType

    fields = {
        "changes",
        "operation",
        "old_positions",
        "new_positions",
        "old_pages",
        "new_pages",
        "explanation",
        "uncertain",
        "issues",
    }
    result: dict[str, Any] = {
        "finish_reason": finish_reason
        if finish_reason in ("stop", "length", "content_filter", "tool_calls", None)
        else "other",
        "errors": [],
        "fields": [],
        "unknown_fields": 0,
        "cardinalities": {},
        "change_shapes": [],
    }
    try:
        payload = json.loads(content)
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, dict):
        result["fields"] = sorted(set(payload) & fields)
        result["unknown_fields"] = min(len(set(payload) - fields), 1000)
        result["cardinalities"] = {
            key: min(len(value), 10000)
            for key, value in payload.items()
            if key in fields and isinstance(value, list)
        }
        changes = payload.get("changes")
        if isinstance(changes, list):
            for change in changes[:20]:
                if not isinstance(change, dict):
                    result["change_shapes"].append({"object": False})
                    continue
                operation = change.get("operation")
                result["change_shapes"].append(
                    {
                        "object": True,
                        "fields": sorted(set(change) & fields),
                        "unknown_fields": min(len(set(change) - fields), 1000),
                        "operation": operation
                        if isinstance(operation, str)
                        and operation
                        in {
                            "replace",
                            "insert",
                            "remove",
                            "move",
                            "split",
                            "merge",
                            "visual",
                        }
                        else "other",
                        **{
                            key: min(len(change[key]), 10000)
                            if isinstance(change.get(key), list)
                            else None
                            for key in ("old_positions", "new_positions")
                        },
                    }
                )
    if error is not None:
        for detail in error.errors(
            include_url=False, include_context=True, include_input=False
        )[:20]:
            validator = None
            cause = detail.get("ctx", {}).get("error")
            if isinstance(cause, ValueError):
                if str(cause).startswith("invalid_operation_shape:"):
                    validator = "invalid_operation_shape"
                elif (
                    str(cause)
                    == "overlapping change references: each physical change must appear once"
                ):
                    validator = "overlapping_change_references"
            result["errors"].append(
                {
                    "loc": [
                        item
                        if (type(item) is int and 0 <= item <= 10000)
                        or (isinstance(item, str) and item in fields)
                        else "unknown"
                        for item in detail["loc"][:8]
                    ],
                    "type": detail["type"]
                    if detail["type"] in get_args(ErrorType)
                    else "other",
                    "validator": validator,
                }
            )
    return result


def replay_batch47_comparison(report: dict[str, Any] | None = None) -> dict[str, Any]:
    """One pinned, read-only diagnostic; the deployed production comparator is unchanged."""
    import base64
    import hashlib
    import time
    from typing import cast
    from unittest.mock import patch
    from uuid import UUID

    from onyx.db.amendment_sources import get_source_package
    from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
    from onyx.db.regulatory_amendments import get_batch
    from onyx.db.regulatory_annex_changes import list_annex_changes
    from onyx.db.regulatory_annex_dev_cutover import configured_indices
    from onyx.file_store.file_store import get_default_file_store
    from onyx.llm.factory import get_default_llm_with_vision
    from onyx.regulatory import structured_llm
    from onyx.regulatory.amendments.annexes import comparison
    from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
    from onyx.regulatory.amendments.annexes.evidence import validate_evidence_view
    from onyx.regulatory.amendments.annexes.models import (
        AnnexChangeDraft,
        AnnexRenderedPage,
    )
    from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable
    from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR

    report = (
        report
        if report is not None
        else {
            "stage": "batch47_comparison",
            "status": "scope_refused",
            "database_read_only": True,
            "attempts": [],
        }
    )
    if (
        os.environ.get("POSTGRES_DB") != "customs-regulations-dev"
        or os.environ.get("REGULATORY_ANNEX_ENVIRONMENT") != "dev"
    ):
        return report
    set_is_ee_based_on_env_variable()
    CURRENT_TENANT_ID_CONTEXTVAR.set("public")
    configured_indices()
    scope = {
        "batch_id": 47,
        "document_set_id": 21,
        "user_file_id": "e0acea4f-30f6-4b05-9bbf-5909b82ef0ab",
        "created_by": "7e0d56bc-6f9c-4cec-b29b-5a8c9ec2b844",
        "environment": "dev",
    }
    package_id = UUID("359c0315-e8ff-4233-8c53-fb5d17cc16ef")
    review_hash = "92cfd62f7e29917003afd9a1b76be662d758903f57a5c3f94f02c8d50585cd08"
    source_hashes = {
        "old": "2a3035058c5a11366604b7e89052a44a7085eb54c705fd3cf09edc1c69f14ffe",
        "new": "c5a20c8fd76d3dbd4983ceeaacc44eef217716427adec8716475ac5a753a0557",
    }
    with SqlEngine.scoped_engine(
        pool_size=2,
        max_overflow=0,
        connect_args={
            "options": "-c default_transaction_read_only=on",
            "connect_timeout": 10,
        },
    ):
        with get_session_with_current_tenant() as session:
            batch = get_batch(session, 47)
            package = get_source_package(
                session, package_id=package_id, document_set_id=21, environment="dev"
            )
            reviews = list_annex_changes(session, 47)
            if batch is None or package is None or len(reviews) != 1:
                return report
            row = reviews[0]
            if (
                batch.document_set_id != 21
                or batch.source_package_id != package_id
                or str(batch.created_by) != scope["created_by"]
                or str(package.created_by) != scope["created_by"]
                or package.idempotency_key
                != "annex-canary-f5c14cd7-fae3-4b13-a67b-15e7eda68c92"
                or row.batch_id != 47
                or row.environment != "dev"
                or str(row.user_file_id) != scope["user_file_id"]
                or row.review_revision != 1
                or row.publication_generation != 0
                or row.status != "blocked"
                or row.review_sha256 != review_hash
                or context_hash(row.review_payload) != review_hash
            ):
                return report
            draft = AnnexChangeDraft.model_validate(row.review_payload)
            if (
                draft.source_package_id != package_id
                or draft.user_file_id != row.user_file_id
                or package.manifest_sha256 != draft.source_manifest_sha256
                or draft.source_manifest_sha256
                != "1679e807427c324ed9c98bc643cd4f648e7055be28defa4e5f91b93042dee16f"
                or draft.corrections
            ):
                return report
        # No session spans model calls. The scoped engine also governs FileStore/factory reads.
        store = get_default_file_store()
        sides: dict[str, Any] = {}
        frozen_hashes: list[str] = []
        for side, extraction in (
            ("old", draft.old_extraction),
            ("new", draft.new_extraction),
        ):
            if (
                extraction is None
                or extraction.evidence_view is None
                or validate_evidence_view(extraction)
            ):
                return report
            view = extraction.evidence_view
            if len(view.parents) != 1 or view.parents[0].sha256 != source_hashes[side]:
                return report
            pages = []
            for mapping in sorted(view.pages, key=lambda item: item.view_page):
                matching = [
                    item
                    for item in draft.evidence
                    if item.side == side
                    and item.kind == "comparison_page"
                    and item.parent_sha256 == source_hashes[side]
                    and item.parent_file_id == view.parents[0].file_id
                    and item.locator.page == mapping.original_page
                ]
                if len(matching) != 1:
                    return report
                item = matching[0]
                record = store.read_file_record(item.file_id)
                if (
                    not isinstance(record.file_metadata, dict)
                    or cast(dict[str, Any], record.file_metadata).get(
                        "annex_review_scope"
                    )
                    != scope
                ):
                    return report
                with store.read_file(item.file_id) as stream:
                    content = stream.read(25 * 1024 * 1024 + 1)
                if (
                    len(content) != item.byte_count
                    or len(content) > 25 * 1024 * 1024
                    or hashlib.sha256(content).hexdigest() != item.sha256
                    or item.locator.original_width is None
                    or item.locator.original_height is None
                ):
                    return report
                pages.append(
                    AnnexRenderedPage(
                        page=mapping.view_page,
                        width=item.locator.original_width,
                        height=item.locator.original_height,
                        png=content,
                    )
                )
                frozen_hashes.append(item.sha256)
            sides[side] = (extraction, pages)
        llm = get_default_llm_with_vision()
        if (
            llm is None
            or llm.config.model_provider != "vertex_ai"
            or draft.comparison is None
            or draft.comparison.model_snapshot is None
            or draft.comparison.model_snapshot.model_provider
            != llm.config.model_provider
            or draft.comparison.model_snapshot.model_name != llm.config.model_name
        ):
            report["status"] = "provider_refused"
            return report
        report.update(
            {
                "model_provider": "vertex_ai",
                "model_name": llm.config.model_name
                if re.fullmatch(r"gemini-[a-z0-9.-]{1,60}", llm.config.model_name)
                else "other",
                "model_snapshot_sha256": context_hash(
                    {
                        "model_provider": llm.config.model_provider,
                        "model_name": llm.config.model_name,
                    }
                ),
                "review_sha256": review_hash,
                "page_sha256": frozen_hashes,
                "old_extraction_sha256": context_hash(
                    sides["old"][0].model_dump(mode="json")
                ),
                "new_extraction_sha256": context_hash(
                    sides["new"][0].model_dump(mode="json")
                ),
            }
        )
        real_invoke, real_validate, real_generate = (
            llm.invoke,
            structured_llm._validate_json_object,
            comparison.generate_structured,
        )
        finish_reason: Any = None

        def observe_invoke(*args: Any, **kwargs: Any) -> Any:
            nonlocal finish_reason
            report["provider_attempt_count"] = (
                report.get("provider_attempt_count", 0) + 1
            )
            response = real_invoke(*args, **kwargs)
            finish_reason = response.choice.finish_reason
            return response

        def observe_validate(content: str, model: Any) -> Any:
            from pydantic import ValidationError

            try:
                result = real_validate(content, model)
            except ValidationError as error:
                report["attempts"].append(
                    comparison_diagnostic_summary(
                        structured_llm._extract_json_object(content),
                        error,
                        finish_reason,
                    )
                )
                raise
            report["attempts"].append(
                comparison_diagnostic_summary(
                    structured_llm._extract_json_object(content), None, finish_reason
                )
            )
            return result

        def observe_generate(*args: Any, **kwargs: Any) -> Any:
            if draft.comparison is None:
                raise ValueError("retained_comparison_required")
            sent = [
                hashlib.sha256(
                    base64.b64decode(item.image_url.url.split(",", 1)[1], validate=True)
                ).hexdigest()
                for item in kwargs["image_parts"]
            ]
            if sent != [item.sha256 for item in draft.comparison.image_manifest]:
                raise ValueError("retained_comparison_images_differ")
            return real_generate(*args, **kwargs, deadline=time.monotonic() + 150)

        with (
            patch.object(llm, "invoke", observe_invoke),
            patch.object(structured_llm, "_validate_json_object", observe_validate),
            patch.object(comparison, "generate_structured", observe_generate),
        ):
            result = comparison.compare_annexes(
                old=sides["old"][0],
                new=sides["new"][0],
                old_pages=sides["old"][1],
                new_pages=sides["new"][1],
                llm=llm,
                instruction="\n\n".join(draft.instruction_texts),
            )
        report.update(
            {
                "status": "ready" if result.ready else "blocked",
                "change_count": len(result.changes),
                "invalid_proposal": "invalid_comparison_proposal" in result.issues,
            }
        )
    return report


def diagnose_batch47_comparison(driver: Driver, pod: str, container: str) -> None:
    if driver.sha != BATCH47_RUNTIME:
        return
    program = (
        """
import contextlib, io, json, logging, os, re, signal
from typing import Any
logging.disable(logging.CRITICAL)
def expired(*args):
    raise TimeoutError()
signal.signal(signal.SIGALRM, expired)
signal.alarm(180)
"""
        + inspect.getsource(comparison_diagnostic_summary)
        + "\n"
        + inspect.getsource(replay_batch47_comparison)
        + """
report = {"stage": "batch47_comparison", "status": "scope_refused", "database_read_only": True, "attempts": []}
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try:
        replay_batch47_comparison(report)
    except Exception as error:
        from onyx.regulatory.amendments.annexes.dev_acceptance import safe_failure_detail
        report.update({"status": "failed", "failure_detail": safe_failure_detail("source_review", error)})
print(json.dumps(report, sort_keys=True))
"""
    )
    output = driver.command(
        [
            "kubectl",
            "--namespace",
            NAMESPACE,
            "exec",
            pod,
            "-c",
            container,
            "--",
            "sh",
            "-eu",
            "-c",
            ". /vault/secrets/config; export PGOPTIONS='-c default_transaction_read_only=on'; exec python -c \"$1\"",
            "batch47-comparison",
            program,
        ],
        timeout=200,
    )
    if len(output.encode()) > 24000:
        raise CutoverRefusal("fixed_diagnostic_report_required")
    report = json.loads(output)
    if (
        not isinstance(report, dict)
        or report.get("stage") != "batch47_comparison"
        or report.get("status")
        not in {"scope_refused", "provider_refused", "failed", "ready", "blocked"}
    ):
        raise CutoverRefusal("fixed_diagnostic_report_required")
    allowed_keys = {
        "stage",
        "status",
        "database_read_only",
        "provider_attempt_count",
        "attempts",
        "model_provider",
        "model_name",
        "model_snapshot_sha256",
        "review_sha256",
        "page_sha256",
        "old_extraction_sha256",
        "new_extraction_sha256",
        "change_count",
        "invalid_proposal",
        "failure_detail",
        "finish_reason",
        "errors",
        "fields",
        "unknown_fields",
        "cardinalities",
        "change_shapes",
        "object",
        "operation",
        "old_positions",
        "new_positions",
        "old_pages",
        "new_pages",
        "changes",
        "explanation",
        "uncertain",
        "issues",
        "loc",
        "type",
        "validator",
    }
    allowed_words = allowed_keys | {
        "batch47_comparison",
        "scope_refused",
        "provider_refused",
        "failed",
        "ready",
        "blocked",
        "vertex_ai",
        "stop",
        "length",
        "content_filter",
        "tool_calls",
        "other",
        "unknown",
        "replace",
        "insert",
        "remove",
        "move",
        "split",
        "merge",
        "visual",
        "invalid_operation_shape",
        "overlapping_change_references",
        "missing",
        "extra_forbidden",
        "list_type",
        "int_type",
        "int_parsing",
        "greater_than_equal",
        "bool_type",
        "bool_parsing",
        "literal_error",
        "string_type",
        "json_invalid",
        "model_type",
        "model_attributes_type",
        "value_error",
        "too_long",
    }
    pending = [report]
    visited = 0
    while pending:
        value = pending.pop()
        visited += 1
        if visited > 3000:
            raise CutoverRefusal("fixed_diagnostic_report_required")
        if isinstance(value, dict):
            if set(value) - allowed_keys:
                raise CutoverRefusal("fixed_diagnostic_report_required")
            for key, child in value.items():
                if key == "failure_detail":
                    if not isinstance(child, str) or len(child) > 4000:
                        raise CutoverRefusal("fixed_diagnostic_report_required")
                    continue
                pending.append(child)
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str):
            if value not in allowed_words and not re.fullmatch(
                r"[0-9a-f]{64}|gemini-[a-z0-9.-]{1,60}", value
            ):
                raise CutoverRefusal("fixed_diagnostic_report_required")
        elif value is not None and type(value) not in (int, bool):
            raise CutoverRefusal("fixed_diagnostic_report_required")
    print(json.dumps(report, sort_keys=True), flush=True)


SOURCE861_RUNTIME = "8612398f20e5d3d03d154fffa196c5bf951b1862"


def source861_scope_matches(run: Any, package: Any, scope: Any) -> bool:
    from datetime import datetime, timedelta

    expected_created = datetime.fromisoformat("2026-09-11T15:56:39.875587+00:00")
    return bool(
        run.release_sha == "8612398f20e5d3d03d154fffa196c5bf951b1862"
        and str(run.run_id) == "5950d8bc-5dac-443b-b143-8494d0b082d1"
        and str(run.file_id) == "a3e6717c-ac94-4a05-a04c-16302f0f2f75"
        and run.document_set_id == 23
        and str(run.package_id) == "4ddd28f3-a9a5-44de-be92-f871fb17469f"
        and run.batch_id is None
        and run.phase == "cleaned"
        and package is not None
        and str(package.id) == str(run.package_id)
        and package.document_set_id == 23
        and package.environment == "dev"
        and package.created_by == run.user_id
        and package.idempotency_key == "annex-canary-" + str(run.run_id)
        and run.created_at == expected_created
        and run.created_at
        <= package.created_at
        <= run.created_at + timedelta(seconds=720)
        and scope is not None
        and not scope.is_public
        and scope.user_id == run.user_id
        and scope.name == run.name
    )


def source861_issue_codes(issues: Any) -> list[str]:
    allowed = {
        "acquisition_failed",
        "404",
        "archive_limit",
        "asset_byte_limit",
        "asset_count_limit",
        "depth_limit",
        "download_failed",
        "download_time_limit",
        "empty_source",
        "encrypted_pdf",
        "mime_mismatch",
        "missing_embedded_source",
        "missing_internal_target",
        "missing_source",
        "package_byte_limit",
        "page_limit",
        "parse_failed",
        "parse_time_limit",
        "redirect_loop",
        "relationship_limit",
        "text_limit",
        "time_limit",
        "truncated",
        "unsupported_format",
        "missing_base_url",
        "unsafe_url",
    }
    if not isinstance(issues, list) or len(issues) > 100:
        return ["unknown_issue"]
    return [
        item.get("code")
        if isinstance(item, dict) and item.get("code") in allowed
        else "unknown_issue"
        for item in issues
    ]


def source861_failure(report: dict[str, Any], error: BaseException) -> None:
    from typing import get_args

    from pydantic import ValidationError
    from pydantic_core import ErrorType

    from onyx.regulatory.amendments.annexes.dev_acceptance import safe_failure_detail

    known = {
        "pdf_visual_extraction_incomplete",
        "pdf_visual_extraction_uncertain",
        "pdf_table_row_ambiguous",
        "pdf_table_cells_overlap",
        "pdf_vision_preparation_deadline",
        "fixed_pdf_evidence_limit",
        "source_diagnostic_attempt_limit",
        "source_diagnostic_http_limit",
    }
    report["status"] = "reproduction_failed"
    report["failure_detail"] = safe_failure_detail("pdf_vision", error)
    report["failure_code"] = (
        error.args[0]
        if (
            type(error) in {ValueError, TimeoutError}
            and len(error.args) == 1
            and isinstance(error.args[0], str)
            and error.args[0] in known
        )
        else "unknown"
    )
    report["schema_errors"] = []
    fields = {
        "elements",
        "kind",
        "text",
        "normalized_box",
        "box",
        "table_role",
        "status",
        "issues",
    }
    seen: set[int] = set()
    while error is not None and id(error) not in seen and len(seen) < 8:
        seen.add(id(error))
        if isinstance(error, ValidationError):
            report["schema_errors"] = [
                {
                    "type": entry["type"]
                    if entry["type"] in get_args(ErrorType)
                    else "unknown",
                    "loc": [
                        part
                        if (type(part) is int and 0 <= part <= 10000)
                        or (isinstance(part, str) and part in fields)
                        else "unknown"
                        for part in entry["loc"][:8]
                    ],
                }
                for entry in error.errors(include_input=False, include_context=False)[
                    :20
                ]
            ]
            break
        next_error = error.__cause__ or error.__context__
        if next_error is None:
            break
        error = next_error


def reproduce_source861(report: dict[str, Any]) -> None:
    import hashlib
    from typing import cast
    from unittest.mock import patch
    from uuid import UUID

    import httpx
    import litellm

    from onyx.db.amendment_sources import get_source_package
    from onyx.db.document_set import get_document_set_by_id
    from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
    from onyx.db.regulatory_annex_acceptance import verify_dev_configuration
    from onyx.file_store.file_store import FileStore, get_default_file_store
    from onyx.llm.factory import get_default_llm_with_vision
    from onyx.regulatory.amendments import pdf_vision
    from onyx.regulatory.amendments.annexes import extraction
    from onyx.regulatory.amendments.annexes.acceptance_pdf_vision import (
        MemoryEvidenceStore,
    )
    from onyx.regulatory.amendments.annexes.dev_acceptance import refuse_fetch
    from onyx.regulatory.amendments.annexes.sources import (
        MAX_PACKAGE_SECONDS,
        acquire_source_package,
    )
    from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable
    from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR

    if (
        os.environ.get("POSTGRES_DB") != "customs-regulations-dev"
        or os.environ.get("REGULATORY_ANNEX_ENVIRONMENT") != "dev"
        or os.environ.get("PGOPTIONS") != "-c default_transaction_read_only=on"
    ):
        report["scope_failure"] = "env"
        return
    set_is_ee_based_on_env_variable()
    CURRENT_TENANT_ID_CONTEXTVAR.set("public")
    configuration = verify_dev_configuration()
    report["configuration_verified"] = True
    with SqlEngine.scoped_engine(
        pool_size=2,
        max_overflow=0,
        connect_args={
            "options": "-c default_transaction_read_only=on",
            "connect_timeout": 10,
        },
    ):
        with get_session_with_current_tenant() as session:
            run = load_source861_canary(session)
            if run is None:
                report["scope_failure"] = "run_missing"
                return
            package = get_source_package(
                session,
                package_id=UUID("4ddd28f3-a9a5-44de-be92-f871fb17469f"),
                document_set_id=23,
                environment="dev",
            )
            scope = get_document_set_by_id(session, 23)
            if not source861_scope_matches(run, package, scope) or package is None:
                report["scope_failure"] = "ownership"
                return
            report.update(
                package_status=package.status,
                issues=source861_issue_codes(package.issues),
                asset_count=package.asset_count,
                manifest_present=bool(
                    package.manifest_file_id or package.manifest_sha256
                ),
            )
            input_file_id, spec = package.input_file_id, dict(package.input_spec)
        if (
            report["issues"] != ["acquisition_failed"]
            or report["manifest_present"]
            or report["package_status"] != "failed"
            or report["asset_count"] != 0
        ):
            report["status"] = "stored_issues_only"
            return
        if (
            not input_file_id
            or spec.get("mime_type") != "application/pdf"
            or spec.get("url")
            or spec.get("base_url")
        ):
            report["scope_failure"] = "source_spec"
            return
        with get_default_file_store().read_file(input_file_id) as stream:
            content = stream.read(25 * 1024 * 1024 + 1)
        expected = "c5a20c8fd76d3dbd4983ceeaacc44eef217716427adec8716475ac5a753a0557"
        if hashlib.sha256(content).hexdigest() != expected:
            report["scope_failure"] = "source_hash"
            return
        report["original_sha256"] = expected
        if configuration["vision_provider"] != "vertex_ai":
            report["status"] = "provider_refused"
            return
        llm = get_default_llm_with_vision()
        if llm is None or llm.config.model_provider != "vertex_ai":
            report["status"] = "provider_refused"
            return
        report["model_provider"] = "vertex_ai"
        report["model_snapshot_sha256"] = hashlib.sha256(
            llm.config.model_dump_json().encode()
        ).hexdigest()
        report["reproduction"] = True
        deadline = time.monotonic() + MAX_PACKAGE_SECONDS
        result = acquire_source_package(
            content=content,
            mime_type="application/pdf",
            display_name=spec.get("display_name", "source"),
            fetch=refuse_fetch,
        )
        report["acquisition_issues"] = source861_issue_codes(
            [item.model_dump() for item in result.issues]
        )
        if (
            result.status != "ready"
            or result.issues
            or result.links
            or len(result.assets) != 1
            or result.assets[0].sha256 != expected
        ):
            report["status"] = "acquisition_refused"
            return
        memory = MemoryEvidenceStore(content)
        original_generate = extraction.generate_structured
        original_completion, original_send = litellm.completion, httpx.Client.send
        original_transcript = pdf_vision.pdf_transcript
        original_invoke = llm.invoke
        active = False
        page_attempts = 0
        attempt_http_count = 0

        def generate(*args: Any, **kwargs: Any) -> Any:
            nonlocal page_attempts
            report["page_index"] += 1
            page_attempts = 0
            if report["page_index"] > 4:
                raise ValueError("source_diagnostic_attempt_limit")
            return original_generate(*args, **kwargs)

        def invoke(*args: Any, **kwargs: Any) -> Any:
            nonlocal active
            if active:
                raise ValueError("source_diagnostic_attempt_limit")
            active = True
            try:
                return original_invoke(*args, **kwargs)
            finally:
                active = False

        def completion(*args: Any, **kwargs: Any) -> Any:
            nonlocal page_attempts, attempt_http_count
            if (
                not active
                or page_attempts >= 3
                or report["attempt_count"] >= 12
                or time.monotonic() >= deadline
            ):
                raise ValueError("source_diagnostic_attempt_limit")
            page_attempts += 1
            report["attempt_count"] += 1
            attempt_http_count = 0
            kwargs.update(num_retries=0, max_retries=0)
            return original_completion(*args, **kwargs)

        def send(
            client: httpx.Client, request: httpx.Request, **kwargs: Any
        ) -> httpx.Response:
            nonlocal attempt_http_count
            if (
                not active
                or page_attempts == 0
                or attempt_http_count >= 1
                or report["http_request_count"] >= 12
            ):
                raise ValueError("source_diagnostic_http_limit")
            attempt_http_count += 1
            report["http_request_count"] += 1
            return original_send(client, request, **kwargs)

        def transcript(value: Any) -> str:
            report["extraction"] = {
                "page_count": value.page_count,
                "element_count": len(value.elements),
                "issue_count": len(value.issues),
                "uncertain_count": sum(
                    item.status != "readable" for item in value.elements
                ),
                "element_issue_count": sum(
                    bool(item.issues) for item in value.elements
                ),
            }
            return original_transcript(value)

        with (
            patch.object(llm, "invoke", invoke),
            patch.object(extraction, "generate_structured", generate),
            patch.object(litellm, "completion", completion),
            patch.object(httpx.Client, "send", send),
            patch.object(pdf_vision, "pdf_transcript", transcript),
        ):
            prepared = pdf_vision.prepare_pdf_source(
                result.assets[0],
                store=cast(FileStore, memory),
                llm=llm,
                deadline=deadline,
            )
        if prepared.pdf_vision is None:
            raise ValueError("pdf_visual_extraction_incomplete")
        report["transcript_sha256"] = prepared.pdf_vision.transcript_sha256
        report["status"] = "reproduction_passed"


def validate_source861_report(report: Any) -> None:
    keys = {
        "scope_failure",
        "stage",
        "status",
        "database_read_only",
        "reproduction",
        "attempt_count",
        "http_request_count",
        "page_index",
        "package_status",
        "issues",
        "asset_count",
        "manifest_present",
        "configuration_verified",
        "original_sha256",
        "model_provider",
        "model_snapshot_sha256",
        "acquisition_issues",
        "extraction",
        "page_count",
        "element_count",
        "issue_count",
        "uncertain_count",
        "element_issue_count",
        "transcript_sha256",
        "failure_detail",
        "failure_code",
        "schema_errors",
        "type",
        "loc",
    }
    words = {
        "env",
        "run_missing",
        "ownership",
        "source_spec",
        "source_hash",
        "source861",
        "scope_refused",
        "stored_issues_only",
        "provider_refused",
        "acquisition_refused",
        "reproduction_failed",
        "reproduction_passed",
        "unknown",
        "processing",
        "ready",
        "partial",
        "blocked",
        "failed",
        "vertex_ai",
        "pdf_visual_extraction_incomplete",
        "pdf_visual_extraction_uncertain",
        "pdf_table_row_ambiguous",
        "pdf_table_cells_overlap",
        "pdf_vision_preparation_deadline",
        "fixed_pdf_evidence_limit",
        "source_diagnostic_attempt_limit",
        "source_diagnostic_http_limit",
        "elements",
        "kind",
        "text",
        "normalized_box",
        "box",
        "table_role",
        "status",
        "issues",
        "missing",
        "extra_forbidden",
        "list_type",
        "int_type",
        "int_parsing",
        "float_type",
        "float_parsing",
        "tuple_type",
        "string_too_long",
        "string_too_short",
        "greater_than_equal",
        "less_than_equal",
        "bool_type",
        "bool_parsing",
        "literal_error",
        "string_type",
        "json_invalid",
        "model_type",
        "model_attributes_type",
        "value_error",
        "too_long",
        "too_short",
        "finite_number",
        "unknown_issue",
    }
    if (
        not isinstance(report, dict)
        or report.get("stage") != "source861"
        or report.get("database_read_only") is not True
        or report.get("status")
        not in {
            "scope_refused",
            "stored_issues_only",
            "provider_refused",
            "acquisition_refused",
            "reproduction_failed",
            "reproduction_passed",
        }
    ):
        raise CutoverRefusal("fixed_diagnostic_report_required")
    pending = [report]
    visited = 0
    while pending:
        value = pending.pop()
        visited += 1
        if visited > 1500:
            raise CutoverRefusal("fixed_diagnostic_report_required")
        if isinstance(value, dict):
            if set(value) - keys:
                raise CutoverRefusal("fixed_diagnostic_report_required")
            for key, child in value.items():
                if key == "scope_failure" and child not in {
                    "env",
                    "run_missing",
                    "ownership",
                    "source_spec",
                    "source_hash",
                }:
                    raise CutoverRefusal("fixed_diagnostic_report_required")
                if key == "failure_detail":
                    if not isinstance(child, str) or len(child) > 4000:
                        raise CutoverRefusal("fixed_diagnostic_report_required")
                    continue
                if key in {"issues", "acquisition_issues"}:
                    if (
                        not isinstance(child, list)
                        or source861_issue_codes([{"code": code} for code in child])
                        != child
                    ):
                        if child != ["unknown_issue"]:
                            raise CutoverRefusal("fixed_diagnostic_report_required")
                    continue
                pending.append(child)
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str):
            if value not in words and not re.fullmatch("[0-9a-f]{64}", value):
                raise CutoverRefusal("fixed_diagnostic_report_required")
        elif value is not None and not (
            type(value) is bool or type(value) is int and 0 <= value <= 100000
        ):
            raise CutoverRefusal("fixed_diagnostic_report_required")
    for key, maximum in (
        ("attempt_count", 12),
        ("http_request_count", 12),
        ("page_index", 4),
    ):
        if type(report.get(key)) is not int or not 0 <= report[key] <= maximum:
            raise CutoverRefusal("fixed_diagnostic_report_required")


def diagnose_source861(driver: Driver, pod: str, container: str) -> None:
    if driver.sha != SOURCE861_RUNTIME:
        raise CutoverRefusal("fixed_source_runtime_required")
    program = """
import contextlib, io, json, logging, os, signal, time
from typing import Any
from sqlalchemy.orm import Session
from onyx.db.regulatory_annex_acceptance import CanaryRun
logging.disable(logging.CRITICAL)
def expired(*args):
    raise TimeoutError()
signal.signal(signal.SIGALRM, expired)
signal.alarm(220)
"""
    program += (
        "\n"
        + Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("onyx/db/regulatory_annex_acceptance_diagnostic.py")
        .read_text()
    )
    for function in (
        source861_scope_matches,
        source861_issue_codes,
        source861_failure,
        reproduce_source861,
    ):
        program += "\n" + inspect.getsource(function)
    program += """
report = {"stage": "source861", "status": "scope_refused", "database_read_only": True, "reproduction": False, "attempt_count": 0, "http_request_count": 0, "page_index": 0}
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try:
        reproduce_source861(report)
    except Exception as error:
        source861_failure(report, error)
print(json.dumps(report, sort_keys=True))
"""
    output = driver.command(
        [
            "kubectl",
            "--namespace",
            NAMESPACE,
            "exec",
            pod,
            "-c",
            container,
            "--",
            "sh",
            "-eu",
            "-c",
            '. /vault/secrets/config; export PGOPTIONS="-c default_transaction_read_only=on"; exec python -c "$1"',
            "source861-diagnostic",
            program,
        ],
        timeout=240,
    )
    if len(output.encode()) > 16000:
        raise CutoverRefusal("fixed_diagnostic_report_required")
    report = json.loads(output)
    validate_source861_report(report)
    print(json.dumps(report, sort_keys=True), flush=True)


CHAT50792_RUNTIME = "50792ae3d877577c4dafcf577bc0027b593370d9"


def validate_chat50792_report(report: Any) -> None:
    keys = {
        "stage",
        "status",
        "database_read_only",
        "configuration_verified",
        "scope_failure",
        "assistant_id",
        "assistant_present",
        "assistant_scope_count",
        "assistant_owned_scope_included",
        "default_present",
        "default_scope_count",
        "default_owned_scope_included",
        "chat_deleted",
        "messages",
        "tools",
        "message_id",
        "assistant",
        "document_count",
        "owned_document_count",
        "citation_count",
        "has_error",
        "error_category",
        "publication_read_present",
        "publication_finalized",
        "tool_call_id",
        "tool_id",
        "result_count",
        "owned_result_count",
        "failure_type",
    }
    words = {
        "chat50792",
        "read",
        "scope_refused",
        "failed",
        "env",
        "run_missing",
        "run_ownership",
        "chat_ownership",
        "evidence_limit",
        "empty",
        "agent_document_set_scope",
        "document_set_access",
        "publication_changed",
        "search_configuration_missing",
        "no_known_error",
        "ValueError",
        "RuntimeError",
        "TimeoutError",
        "Exception",
    }
    if (
        not isinstance(report, dict)
        or report.get("stage") != "chat50792"
        or report.get("database_read_only") is not True
        or report.get("status") not in {"read", "scope_refused", "failed"}
    ):
        raise CutoverRefusal("fixed_diagnostic_report_required")
    pending = [report]
    visited = 0
    while pending:
        value = pending.pop()
        visited += 1
        if visited > 2000:
            raise CutoverRefusal("fixed_diagnostic_report_required")
        if isinstance(value, dict):
            if set(value) - keys:
                raise CutoverRefusal("fixed_diagnostic_report_required")
            pending.extend(value.values())
        elif isinstance(value, list):
            if len(value) > 64:
                raise CutoverRefusal("fixed_diagnostic_report_required")
            pending.extend(value)
        elif isinstance(value, str):
            if value not in words:
                raise CutoverRefusal("fixed_diagnostic_report_required")
        elif value is not None and not (
            type(value) is bool or type(value) is int and 0 <= value <= 2147483647
        ):
            raise CutoverRefusal("fixed_diagnostic_report_required")


def diagnose_chat50792(driver: Driver, pod: str, container: str) -> None:
    if driver.sha != CHAT50792_RUNTIME:
        raise CutoverRefusal("fixed_chat_runtime_required")
    program = """
import contextlib, io, json, logging, os, signal
logging.disable(logging.CRITICAL)
def expired(*args):
    raise TimeoutError()
signal.signal(signal.SIGALRM, expired)
signal.alarm(60)
"""
    program += (
        "\n"
        + Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("onyx/db/regulatory_annex_acceptance_diagnostic.py")
        .read_text()
    )
    program += """
report = {"stage": "chat50792", "status": "scope_refused", "scope_failure": "env", "database_read_only": True}
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try:
        if (os.environ.get("POSTGRES_DB") == "customs-regulations-dev"
            and os.environ.get("REGULATORY_ANNEX_ENVIRONMENT") == "dev"
            and os.environ.get("PGOPTIONS") == "-c default_transaction_read_only=on"):
            from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable
            from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR
            from onyx.db.regulatory_annex_dev_cutover import configured_indices
            from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
            set_is_ee_based_on_env_variable()
            CURRENT_TENANT_ID_CONTEXTVAR.set("public")
            configured_indices()
            report.pop("scope_failure")
            report["configuration_verified"] = True
            with SqlEngine.scoped_engine(pool_size=2, max_overflow=0, connect_args={"options": "-c default_transaction_read_only=on", "connect_timeout": 10}):
                with get_session_with_current_tenant() as session:
                    report.update(load_chat50792_diagnostic(session))
    except Exception as error:
        name = type(error).__name__
        report.update(status="failed", failure_type=name if name in {"ValueError", "RuntimeError", "TimeoutError"} else "Exception")
print(json.dumps(report, sort_keys=True))
"""
    output = driver.command(
        [
            "kubectl",
            "--namespace",
            NAMESPACE,
            "exec",
            pod,
            "-c",
            container,
            "--",
            "sh",
            "-eu",
            "-c",
            '. /vault/secrets/config; export PGOPTIONS="-c default_transaction_read_only=on"; exec python -c "$1"',
            "chat50792-diagnostic",
            program,
        ],
        timeout=80,
    )
    if len(output.encode()) > 24000:
        raise CutoverRefusal("fixed_diagnostic_report_required")
    report = json.loads(output)
    validate_chat50792_report(report)
    print(json.dumps(report, sort_keys=True), flush=True)


MARKDOWN_A8_RUNTIME = "a8a1406d4d4d359298247949bb0e3190f3069598"


def markdown_worker_health() -> dict[str, object]:
    import subprocess

    names = (
        "celery_worker_user_file_processing",
        "celery_worker_regulatory_indexing",
        "celery_beat_regulatory_indexing",
    )
    try:
        result = subprocess.run(
            [
                "supervisorctl",
                "-c",
                "/etc/supervisor/conf.d/supervisord.conf",
                "status",
                *names,
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if len(result.stdout) > 4096:
            return {"worker_health_status": "unavailable"}
        states = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if (
                len(parts) < 2
                or parts[0] not in names
                or parts[0] in states
                or parts[1]
                not in {
                    "STOPPED",
                    "STARTING",
                    "RUNNING",
                    "BACKOFF",
                    "STOPPING",
                    "EXITED",
                    "FATAL",
                    "UNKNOWN",
                }
            ):
                return {"worker_health_status": "unavailable"}
            states[parts[0]] = parts[1]
        if set(states) != set(names):
            return {"worker_health_status": "unavailable"}
        return {
            "worker_health_status": "read",
            "workers": [
                {"process_name": name, "process_state": states[name]} for name in names
            ],
        }
    except (OSError, subprocess.TimeoutExpired):
        return {"worker_health_status": "unavailable"}


def validate_markdown_progress_report(report: Any) -> None:
    import math

    bools = {
        "database_read_only",
        "configuration_verified",
        "current_batch_indexing",
        "current_deferred_indexing",
        "current_vector_disabled",
        "owner_present",
        "gate_closed",
        "writer_manifest_present",
        "receipt_present",
        "receipt_valid",
        "file_present",
        "has_error_code",
    }
    counts = {
        "canonical_count",
        "temporal_count",
        "temporal_retired_count",
        "job_count",
        "attempt_count",
        "message_id",
    }
    hashes = {
        "receipt_raw_sha256",
        "receipt_canonical_sha256",
        "receipt_generation_sha256",
    }
    dates = {
        "time_sent",
        "first_canonical_at",
        "last_canonical_at",
        "old_chat_created_at",
        "old_chat_updated_at",
        "new_chat_created_at",
        "new_chat_updated_at",
        "first_published_at",
        "last_published_at",
        "next_retry_at",
    }
    enums = {
        "message_type": {
            "system",
            "user",
            "assistant",
            "tool_call_response",
            "user_reminder",
        },
        "stage": {"markdown_a8"},
        "status": {"read", "failed", "scope_refused"},
        "scope_failure": {
            "env",
            "run_missing",
            "run_ownership",
            "private_scope",
            "publication_scope",
            "chat_ownership",
            "file_ownership",
            "evidence_limit",
        },
        "failure_type": {"ValueError", "RuntimeError", "TimeoutError", "Exception"},
        "file_status": {
            "PROCESSING",
            "INDEXING",
            "CHUNKED",
            "COMPLETED",
            "SKIPPED",
            "FAILED",
            "CANCELED",
            "DELETING",
        },
        "job_status": {
            "QUEUED",
            "RUNNING",
            "RETRY_WAIT",
            "SUCCEEDED",
            "FAILED",
            "CANCELLING",
            "CANCELLED",
        },
        "job_stage": {
            "PREPARING",
            "CONTEXT_SUBMIT",
            "CONTEXT_WAIT",
            "CONTEXT_APPLY",
            "EMBEDDING",
            "INDEX_WRITE",
            "VERIFY",
            "PUBLISH",
        },
        "worker_health_status": {"read", "unavailable"},
        "process_name": {
            "celery_worker_user_file_processing",
            "celery_worker_regulatory_indexing",
            "celery_beat_regulatory_indexing",
        },
        "process_state": {
            "STOPPED",
            "STARTING",
            "RUNNING",
            "BACKOFF",
            "STOPPING",
            "EXITED",
            "FATAL",
            "UNKNOWN",
        },
    }
    if (
        not isinstance(report, dict)
        or report.get("stage") != "markdown_a8"
        or report.get("database_read_only") is not True
        or report.get("status") not in enums["status"]
    ):
        raise CutoverRefusal("fixed_diagnostic_report_required")
    pending = [report]
    while pending:
        item = pending.pop()
        for key, value in item.items():
            valid = False
            if key in bools:
                valid = value is None or type(value) is bool
            elif key == "pre_answer_processing_seconds":
                valid = (
                    value is None
                    or type(value) in {int, float}
                    and math.isfinite(value)
                    and 0 <= value <= 86400
                )
            elif key in counts:
                valid = type(value) is int and 0 <= value <= 2147483647
            elif key in hashes:
                valid = (
                    value is None
                    or isinstance(value, str)
                    and re.fullmatch(r"[0-9a-f]{64}", value) is not None
                )
            elif key in dates:
                valid = (
                    value is None
                    or isinstance(value, str)
                    and re.fullmatch(
                        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?\+00:00",
                        value,
                    )
                    is not None
                )
            elif key in enums:
                valid = value is None or isinstance(value, str) and value in enums[key]
            elif key in {"jobs", "workers", "old_chat_messages", "new_chat_messages"}:
                valid = (
                    isinstance(value, list)
                    and len(value) <= 64
                    and all(
                        isinstance(row, dict)
                        and not (
                            {
                                "jobs",
                                "workers",
                                "old_chat_messages",
                                "new_chat_messages",
                            }
                            & set(row)
                        )
                        for row in value
                    )
                )
                if valid:
                    pending.extend(value)
            if not valid:
                raise CutoverRefusal("fixed_diagnostic_report_required")


def diagnose_markdown_progress(driver: Driver, pod: str, container: str) -> None:
    if driver.sha != MARKDOWN_A8_RUNTIME:
        raise CutoverRefusal("fixed_markdown_runtime_required")
    program = "import contextlib, io, json, logging, signal\nlogging.disable(logging.CRITICAL)\ndef expired(*args):\n    raise TimeoutError()\nsignal.signal(signal.SIGALRM, expired)\nsignal.alarm(60)\n"
    program += (
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("onyx/db/regulatory_markdown_progress_diagnostic.py")
        .read_text()
    )
    program += "\n" + inspect.getsource(markdown_worker_health)
    program += """
report = {"stage": "markdown_a8", "status": "failed", "database_read_only": True}
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try:
        report.update(read_markdown_progress())
        if report.get("status") == "read":
            report.update(markdown_worker_health())
    except Exception as error:
        name = type(error).__name__
        report.update(status="failed", failure_type=name if name in {"ValueError", "RuntimeError", "TimeoutError"} else "Exception")
print(json.dumps(report, sort_keys=True))
"""
    output = driver.command(
        [
            "kubectl",
            "--namespace",
            NAMESPACE,
            "exec",
            pod,
            "-c",
            container,
            "--",
            "sh",
            "-eu",
            "-c",
            '. /vault/secrets/config; export PGOPTIONS="-c default_transaction_read_only=on"; exec python -c "$1"',
            "markdown-a8-diagnostic",
            program,
        ],
        timeout=80,
    )
    if len(output.encode()) > 24000:
        raise CutoverRefusal("fixed_diagnostic_report_required")
    report = json.loads(output)
    validate_markdown_progress_report(report)
    print(json.dumps(report, sort_keys=True), flush=True)


MARKDOWN_WORKER_RUNTIME = "c3d077dde1cd10c8f9c4a25288fad9fa84f62e9a"


def summarize_markdown_worker(
    node: str, supervisor_pid: int | None, replies: dict[str, Any]
) -> dict[str, str | bool | int]:
    """Only current local worker contract flags and counts may leave the pod."""
    result: dict[str, str | bool | int] = {
        "stage": "markdown_worker",
        "status": "read",
        "supervisor_running": supervisor_pid is not None,
    }
    values: dict[str, Any] = {}
    for name in ("registered", "active_queues", "stats", "active", "reserved"):
        reply = replies.get(name)
        value = reply.get(node) if isinstance(reply, dict) else None
        available = isinstance(value, dict if name == "stats" else list)
        result[name + "_response"] = available
        if not available:
            result["status"] = "unavailable"
            value = {} if name == "stats" else []
        assert isinstance(value, (dict, list))
        if len(value) > (10000 if name == "registered" else 1000):
            raise ValueError("worker_reply_bound_exceeded")
        values[name] = value
    registered = {
        item.split(" ", 1)[0] for item in values["registered"] if isinstance(item, str)
    }
    result["process_registered"] = "process_single_user_file" in registered
    result["index_registered"] = "index_single_user_file" in registered
    queues = {
        item["name"]
        for item in values["active_queues"]
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    result["queues_match"] = queues == {
        "user_file_processing",
        "user_file_project_sync",
        "user_file_delete",
        "user_file_port",
    }
    result["queue_count"] = len(queues)
    stats = values["stats"]
    pid = stats.get("pid")
    result["stats_pid_matches"] = type(pid) is int and pid > 0 and pid == supervisor_pid
    pool = stats.get("pool")
    concurrency = pool.get("max-concurrency") if isinstance(pool, dict) else None
    if type(concurrency) is int and 0 < concurrency <= 1000:
        result["concurrency_available"] = True
        result["concurrency"] = concurrency
    else:
        result["concurrency_available"] = False
        result["concurrency"] = 0
    for kind in ("active", "reserved"):
        rows = values[kind]
        result[kind + "_count"] = len(rows)
        for label, task in (
            ("index", "index_single_user_file"),
            ("process", "process_single_user_file"),
        ):
            result[kind + "_" + label + "_count"] = sum(
                isinstance(row, dict) and row.get("name") == task for row in rows
            )
        result[kind + "_other_count"] = sum(
            not isinstance(row, dict)
            or row.get("name")
            not in {"index_single_user_file", "process_single_user_file"}
            for row in rows
        )
    return result


def read_markdown_worker() -> dict[str, str | bool | int]:
    import re
    import socket
    import subprocess

    from onyx.background.celery.versioned_apps.client import app
    from onyx.configs.app_configs import CELERY_PRIMARY_WORKER_REQUIRED

    pid = None
    try:
        status = subprocess.run(
            [
                "supervisorctl",
                "-c",
                "/etc/supervisor/conf.d/supervisord.conf",
                "status",
                "celery_worker_user_file_processing",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if len(status.stdout) <= 4096:
            match = re.fullmatch(
                r"celery_worker_user_file_processing\s+RUNNING\s+pid ([1-9][0-9]*),[^\n]*\n?",
                status.stdout,
            )
            if match:
                pid = int(match[1])
    except (OSError, subprocess.TimeoutExpired):
        pass
    node = "user_file_processing@" + socket.gethostname()
    inspector = app.control.inspect(timeout=3, destination=[node])
    replies = {
        "registered": inspector.registered(),
        "active_queues": inspector.active_queues(),
        "stats": inspector.stats(),
        "active": inspector.active(),
        "reserved": inspector.reserved(),
    }
    result = summarize_markdown_worker(node, pid, replies)
    result["current_primary_worker_required"] = CELERY_PRIMARY_WORKER_REQUIRED
    return result


def summarize_markdown_task_log(content: str) -> dict[str, str | bool]:
    import re

    task = "719599a4-cddb-402c-ac52-3c3534581d46"
    file = "32ab4fc6-f7cd-4d51-a34f-1c808760908f"
    events = {
        "received": f"Task process_single_user_file[{task}] received",
        "succeeded": f"Task process_single_user_file[{task}] succeeded",
        "expired": f"Discarding revoked task: process_single_user_file[{task}]",
        "started": f"process_user_file_impl - Starting id={file}",
        "file_missing": f"process_user_file_impl - UserFile not found id={file}",
        "lock_held": f"process_user_file_impl - Lock held, skipping user_file_id={file}",
        "failed": f"process_user_file_impl - Error processing file id={file}",
    }
    lines = re.sub(r"\x1b\[[0-9;]*m", "", content).splitlines()
    result: dict[str, str | bool] = {
        "owned_task_" + key: any(value in line for line in lines)
        for key, value in events.items()
    }
    frames: list[str] = []
    for position, line in enumerate(lines):
        if events["failed"] not in line:
            continue
        for following in lines[position + 1 : position + 101]:
            if following.startswith("Traceback "):
                continue
            if following and not following.startswith(" "):
                break
            match = re.fullmatch(
                r'  File "[^"\n]*?((?:onyx|shared_configs|ee/onyx)/[A-Za-z0-9_./-]+\.py)", line ([1-9][0-9]*), in ([A-Za-z0-9_<>]+)',
                following,
            )
            if match and ".." not in match[1].split("/"):
                frames.append(":".join(match.groups()))
    result["owned_task_frames"] = ";".join(frames[:30])
    return result


def read_markdown_task_receipt() -> dict[str, str | bool | int]:
    import os
    import re
    import socket
    from datetime import datetime

    from onyx.background.celery.versioned_apps.client import app

    result: dict[str, str | bool | int] = {
        "owned_task_log_available": False,
        "owned_task_log_truncated": False,
        "owned_task_log_files": 0,
        "owned_task_log_bytes": 0,
    }
    timestamps: list[datetime] = []
    frames: list[str] = []
    path = "/var/log/onyx/celery_worker_user_file_processing.log"
    for number in range(11):
        try:
            with open(path + ("." + str(number) if number else ""), "rb") as log:
                size = os.fstat(log.fileno()).st_size
                content = log.read(16777216).decode("utf-8", errors="replace")
            result["owned_task_log_available"] = True
            result["owned_task_log_truncated"] = (
                bool(result["owned_task_log_truncated"]) or size > 16777216
            )
            result["owned_task_log_files"] = number + 1
            previous_bytes = result["owned_task_log_bytes"]
            assert type(previous_bytes) is int
            result["owned_task_log_bytes"] = previous_bytes + min(size, 16777216)
            snapshot = summarize_markdown_task_log(content)
            for key, value in snapshot.items():
                if isinstance(value, bool):
                    result[key] = bool(result.get(key)) or value
                elif value:
                    frames.append(value)
            matches = list(
                re.finditer(
                    r"[0-9]{2}/[0-9]{2}/[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} [AP]M",
                    content,
                )
            )
            for match in matches[:1] + matches[-1:]:
                timestamps.append(datetime.strptime(match[0], "%m/%d/%Y %I:%M:%S %p"))
        except FileNotFoundError:
            break
    result["owned_task_frames"] = ";".join(";".join(frames).split(";")[:30])
    if timestamps:
        result["owned_log_first_timestamp"] = min(timestamps).isoformat()
        result["owned_log_last_timestamp"] = max(timestamps).isoformat()
    # Use actual queue membership; consumer node names are not an ownership boundary.
    inspector = app.control.inspect(timeout=3)
    replies = inspector.active_queues() or {}
    if len(replies) > 64:
        raise ValueError("consumer_reply_bound_exceeded")
    nodes = sorted(
        node
        for node, queues in replies.items()
        if isinstance(node, str)
        and re.fullmatch(r"[A-Za-z0-9_@.:-]{1,256}", node)
        and isinstance(queues, list)
        and any(
            isinstance(queue, dict) and queue.get("name") == "user_file_processing"
            for queue in queues
        )
    )
    local = "user_file_processing@" + socket.gethostname()
    result["normal_worker_responses"] = len(replies)
    result["normal_worker_other_consumers"] = sum(node != local for node in nodes)
    result["queue_consumer_names"] = ",".join(nodes)
    task = "719599a4-cddb-402c-ac52-3c3534581d46"
    queried = inspector.query_task(task) or {}
    if len(queried) > 64:
        raise ValueError("task_reply_bound_exceeded")
    receivers = []
    for node, tasks in queried.items():
        entry = tasks.get(task) if isinstance(tasks, dict) else None
        if (
            isinstance(node, str)
            and re.fullmatch(r"[A-Za-z0-9_@.:-]{1,256}", node)
            and isinstance(entry, (list, tuple))
            and len(entry) == 2
            and entry[0] in {"active", "reserved", "ready"}
        ):
            receivers.append(node + ":" + entry[0])
    result["owned_task_receivers"] = ",".join(sorted(receivers))
    return result


def read_markdown_delivery() -> dict[str, str | bool | int]:
    import hashlib
    import json

    from onyx.background.celery.celery_redis import (
        celery_get_broker_client,
        celery_get_queue_length,
    )
    from onyx.background.celery.versioned_apps.client import app
    from onyx.configs import app_configs

    with app.connection_for_write() as connection:
        identity = {
            "host": connection.hostname,
            "port": connection.port,
            "database": connection.virtual_host,
            "transport": connection.transport_cls,
            "priority_steps": app.conf.broker_transport_options.get("priority_steps"),
            "separator": app.conf.broker_transport_options.get("sep"),
            "key_prefix": app.conf.broker_transport_options.get("global_keyprefix"),
        }
    broker = celery_get_broker_client(app)
    try:
        depth = celery_get_queue_length("user_file_processing", broker)
    finally:
        broker.close()
    if type(depth) is not int or not 0 <= depth <= 1000000:
        raise ValueError("queue_depth_bound_exceeded")
    return {
        "broker_target_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest(),
        "processing_queue_depth": depth,
        "current_vector_disabled": app_configs.DISABLE_VECTOR_DB,
        "current_defer_indexing": app_configs.DEFER_USER_FILE_INDEXING,
        "current_batch_indexing": app_configs.REGULATORY_BATCH_INDEXING_ENABLED,
    }


def validate_markdown_worker_report(report: Any) -> None:
    booleans = {
        "database_read_only",
        "configuration_verified",
        "supervisor_running",
        "process_registered",
        "index_registered",
        "queues_match",
        "stats_pid_matches",
        "concurrency_available",
        "current_primary_worker_required",
        "current_vector_disabled",
        "current_defer_indexing",
        "current_batch_indexing",
        "owned_task_log_available",
        "owned_task_log_truncated",
    }
    booleans.update(
        "owned_task_" + name
        for name in (
            "received",
            "succeeded",
            "expired",
            "started",
            "file_missing",
            "lock_held",
            "failed",
        )
    )
    booleans.update(
        name + "_response"
        for name in ("registered", "active_queues", "stats", "active", "reserved")
    )
    counts = {
        "queue_count",
        "concurrency",
        "owned_task_log_files",
        "normal_worker_responses",
        "normal_worker_other_consumers",
    } | {
        kind + suffix
        for kind in ("active", "reserved")
        for suffix in ("_count", "_index_count", "_process_count", "_other_count")
    }
    enums = {
        "stage": {"markdown_worker"},
        "role": {"api", "background"},
        "status": {"read", "unavailable", "failed"},
        "failure_type": {"ValueError", "RuntimeError", "TimeoutError", "Exception"},
    }
    if (
        not isinstance(report, dict)
        or report.get("stage") != "markdown_worker"
        or report.get("database_read_only") is not True
    ):
        raise CutoverRefusal("fixed_worker_report_required")
    for key, value in report.items():
        valid = (
            (key in booleans and type(value) is bool)
            or (
                key == "owned_task_frames"
                and isinstance(value, str)
                and len(value) <= 8192
                and re.fullmatch(
                    r"(?:[A-Za-z0-9_./<>-]+:[0-9]+:[A-Za-z0-9_<>]+;?)*", value
                )
                is not None
            )
            or (
                key in {"queue_consumer_names", "owned_task_receivers"}
                and isinstance(value, str)
                and len(value) <= 8192
                and re.fullmatch(r"[A-Za-z0-9_@.,:-]*", value) is not None
            )
            or (
                key in {"owned_log_first_timestamp", "owned_log_last_timestamp"}
                and isinstance(value, str)
                and re.fullmatch(
                    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}", value
                )
                is not None
            )
            or (
                key == "owned_task_log_bytes"
                and type(value) is int
                and 0 <= value <= 184549376
            )
            or (key in counts and type(value) is int and 0 <= value <= 10000)
            or (
                key == "processing_queue_depth"
                and type(value) is int
                and 0 <= value <= 1000000
            )
            or (
                key == "broker_target_sha256"
                and isinstance(value, str)
                and re.fullmatch(r"[0-9a-f]{64}", value) is not None
            )
            or (key in enums and isinstance(value, str) and value in enums[key])
        )
        if not valid:
            raise CutoverRefusal("fixed_worker_report_required")


def diagnose_markdown_worker(
    driver: Driver, pod: str, container: str, *, role: str = "background"
) -> None:
    if driver.sha != MARKDOWN_WORKER_RUNTIME or role not in {"api", "background"}:
        raise CutoverRefusal("fixed_worker_runtime_required")
    program = "import contextlib, io, json, logging, os, signal\nfrom typing import Any\nlogging.disable(logging.CRITICAL)\ndef expired(*args):\n    raise TimeoutError()\nsignal.signal(signal.SIGALRM, expired)\nsignal.alarm(60)\n"
    program += (
        inspect.getsource(summarize_markdown_worker)
        + "\n"
        + inspect.getsource(read_markdown_worker)
        + "\n"
        + inspect.getsource(read_markdown_delivery)
        + "\n"
        + inspect.getsource(summarize_markdown_task_log)
        + "\n"
        + inspect.getsource(read_markdown_task_receipt)
    )
    program += "\nrole = " + repr(role) + "\n"
    program += """
report = {"stage": "markdown_worker", "status": "failed", "database_read_only": True}
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try:
        if os.environ.get("POSTGRES_DB") != "customs-regulations-dev" or os.environ.get("REGULATORY_ANNEX_ENVIRONMENT") != "dev" or os.environ.get("PGOPTIONS") != "-c default_transaction_read_only=on":
            raise ValueError()
        from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable
        set_is_ee_based_on_env_variable()
        from onyx.db.regulatory_annex_dev_cutover import configured_indices
        configured_indices()
        report["configuration_verified"] = True
        report.update(read_markdown_delivery())
        report["role"] = role
        report["status"] = "read"
        if role == "background":
            report.update(read_markdown_worker())
            report.update(read_markdown_task_receipt())
    except Exception as error:
        name = type(error).__name__
        report.update(status="failed", failure_type=name if name in {"ValueError", "RuntimeError", "TimeoutError"} else "Exception")
print(json.dumps(report, sort_keys=True))
"""
    output = driver.command(
        [
            "kubectl",
            "--namespace",
            NAMESPACE,
            "exec",
            pod,
            "-c",
            container,
            "--",
            "sh",
            "-eu",
            "-c",
            '. /vault/secrets/config; export PGOPTIONS="-c default_transaction_read_only=on"; exec python -c "$1"',
            "markdown-worker-diagnostic",
            program,
        ],
        timeout=80,
    )
    if len(output.encode()) > 8000:
        raise CutoverRefusal("fixed_worker_report_required")
    report = json.loads(output)
    validate_markdown_worker_report(report)
    print(json.dumps(report, sort_keys=True), flush=True)


def diagnose_release(driver: Driver, runner_sha: str) -> None:
    driver.validate_target()
    state = driver.get("configmap", STATE)["data"]
    if state["sha"] != driver.sha or state["phase"] != "released":
        raise CutoverRefusal("successful_matching_cutover_required")
    require_release_runs(driver.sha)
    driver.verify_runtime(readiness=False)
    verify_frontend(driver)
    print(
        json.dumps(
            {"runner_sha": runner_sha, "runtime_sha": driver.sha}, sort_keys=True
        ),
        flush=True,
    )
    pod = driver.pods("background")[0]
    container = next(
        item
        for item in pod["spec"]["containers"]
        if item["image"] == f"{REPOSITORY}:{driver.sha}"
    )
    if driver.sha == MARKDOWN_WORKER_RUNTIME:
        diagnose_markdown_worker(driver, pod["metadata"]["name"], container["name"])
        for api_pod in driver.pods("api"):
            api_container = next(
                item
                for item in api_pod["spec"]["containers"]
                if item["image"] == f"{REPOSITORY}:{driver.sha}"
            )
            diagnose_markdown_worker(
                driver, api_pod["metadata"]["name"], api_container["name"], role="api"
            )
        return
    if driver.sha == MARKDOWN_A8_RUNTIME:
        diagnose_markdown_progress(driver, pod["metadata"]["name"], container["name"])
        return
    if driver.sha == CHAT50792_RUNTIME:
        diagnose_chat50792(driver, pod["metadata"]["name"], container["name"])
        return
    if driver.sha == SOURCE861_RUNTIME:
        diagnose_source861(driver, pod["metadata"]["name"], container["name"])
        return
    if diagnose_batch44_logs(driver.sha) == "application_group_absent":
        diagnose_logging_metadata(driver)
    if driver.sha == BATCH47_RUNTIME:
        diagnose_batch47_comparison(driver, pod["metadata"]["name"], container["name"])
        return
    diagnose_batch44_elasticsearch(driver, pod["metadata"]["name"], container["name"])
    output = driver.command(
        [
            "kubectl",
            "--namespace",
            NAMESPACE,
            "exec",
            pod["metadata"]["name"],
            "-c",
            container["name"],
            "--",
            "sh",
            "-eu",
            "-c",
            '. /vault/secrets/config; exec python -c "$1"',
            "annex-diagnose",
            DIAGNOSTIC_PROGRAM,
        ],
        timeout=180,
    )
    if len(output.encode()) > 64000:
        raise CutoverRefusal("fixed_diagnostic_report_required")
    report = json.loads(output)
    if (
        not isinstance(report, dict)
        or set(report) != {"stages"}
        or not isinstance(report["stages"], list)
        or len(report["stages"]) != 2
    ):
        raise CutoverRefusal("fixed_diagnostic_report_required")
    for raw_item, stage in zip(report["stages"], ("configuration", "native_parser")):
        if not isinstance(raw_item, dict):
            raise CutoverRefusal("fixed_diagnostic_report_required")
        item = cast(dict[str, Any], raw_item)
        if (
            not isinstance(item, dict)
            or set(item) - {"stage", "exception_class", "frames", "configuration"}
            or item.get("stage") != stage
        ):
            raise CutoverRefusal("fixed_diagnostic_report_required")
        exception = item.get("exception_class")
        if exception is not None and (
            not isinstance(exception, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,99}", exception)
        ):
            raise CutoverRefusal("fixed_diagnostic_report_required")
        frames = item.get("frames")
        if not isinstance(frames, list):
            raise CutoverRefusal("fixed_diagnostic_report_required")
        for frame in frames:
            if (
                not isinstance(frame, dict)
                or set(frame) != {"filename", "function", "line"}
                or not isinstance(frame["filename"], str)
                or not re.fullmatch(
                    r"(?:onyx|shared_configs|ee/onyx)/[A-Za-z0-9_./-]+\.py",
                    frame["filename"],
                )
                or ".." in frame["filename"].split("/")
                or not isinstance(frame["function"], str)
                or not re.fullmatch(r"[A-Za-z0-9_<>]+", frame["function"])
                or type(frame["line"]) is not int
                or frame["line"] < 1
            ):
                raise CutoverRefusal("fixed_diagnostic_report_required")
        if "configuration" in item:
            config: dict[str, Any] = item["configuration"]
            if (
                stage != "configuration"
                or exception is not None
                or not isinstance(config, dict)
                or set(config) != {"database", "indices"}
                or config["database"] != "customs-regulations-dev"
                or not isinstance(config["indices"], list)
                or not config["indices"]
                or any(
                    not isinstance(name, str)
                    or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name)
                    or "_dev_" not in name
                    for name in config["indices"]
                )
            ):
                raise CutoverRefusal("fixed_diagnostic_report_required")
    print(json.dumps(report, sort_keys=True), flush=True)


def deploy_same_image(driver: Driver, enabled: bool) -> None:
    render_values(enabled)
    for app in ("background", "api") if enabled else APPS:
        print(
            f"ACTIVATION_STAGE {app}_{'enable' if enabled else 'disable'}", flush=True
        )
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
        if enabled and app == "background":
            print("ACTIVATION_STAGE background_readiness", flush=True)
            driver.verify_app("background")


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
            print("ACTIVATION_STAGE final_runtime", flush=True)
            driver.verify_runtime()
            verify_frontend(driver)
            print("ACTIVATION_STAGE canary", flush=True)
            acceptance(driver, "canary")
        except Exception:
            # Same reviewed binary; disable creation without undoing protected data.
            print("ACTIVATION_STAGE fallback_disable", flush=True)
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
            "annex-diagnose",
        ),
    )
    args = parser.parse_args()
    sha = os.environ.get("IMAGE_TAG", "")
    validate_scope(os.environ.get("env_x", ""), os.environ.get("GITHUB_REF", ""), sha)
    runner_sha = os.environ.get("GITHUB_SHA", "")
    validate_scope(
        os.environ.get("env_x", ""), os.environ.get("GITHUB_REF", ""), runner_sha
    )
    if args.phase != "annex-diagnose" and sha != runner_sha:
        if args.phase in {"annex-verify", "annex-activate"}:
            require_runner_compatibility(sha, runner_sha)
        else:
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
    elif args.phase == "annex-diagnose":
        diagnose_release(driver, runner_sha)
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
