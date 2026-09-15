from email.message import Message
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request

import pytest
from scripts import regulatory_annex_dev_cutover as cutover


def test_fixed_acceptance_passes_validated_release_ownership_metadata() -> None:
    driver = Mock(spec=cutover.Driver)
    driver.sha = "a" * 40
    driver.pods.return_value = [
        {
            "metadata": {"name": "background"},
            "spec": {
                "containers": [
                    {"name": "worker", "image": cutover.REPOSITORY + ":" + driver.sha}
                ]
            },
        }
    ]
    cutover.acceptance(driver, "preflight")
    command = driver.command.call_args.args[0]
    assert "ANNEX_ACCEPTANCE_RELEASE_SHA=" + driver.sha in command
    assert command[-1] == "preflight"


def test_cutover_orders_barrier_before_deployment_and_release() -> None:
    driver = Mock(spec=cutover.Driver)
    driver.continue_protected_release.return_value = False
    cutover.prepare(driver)
    assert [call[0] for call in driver.method_calls] == [
        "validate_target",
        "continue_protected_release",
        "inventory",
        "redis_isolation",
        "create_probe",
        "inspect_indices",
        "record_started",
        "stop_api",
        "drain_background",
        "stop_background",
        "assert_no_writers",
        "block_indices",
        "record_blocked",
        "assert_no_writers",
        "unblock_indices",
        "record_installing",
    ]


def test_failed_barrier_never_restarts_old_writers() -> None:
    driver = Mock(spec=cutover.Driver)
    driver.continue_protected_release.return_value = False
    driver.block_indices.side_effect = RuntimeError("partial acknowledgement")
    with pytest.raises(RuntimeError):
        cutover.prepare(driver)
    assert not driver.record_blocked.called
    assert not driver.unblock_indices.called


def test_release_requires_exact_runtime_before_unblock() -> None:
    driver = Mock(spec=cutover.Driver)
    driver.continue_protected_release.return_value = False
    driver.verify_runtime.side_effect = RuntimeError("wrong queue")
    with pytest.raises(RuntimeError):
        cutover.release(driver)
    assert not driver.unblock_indices.called


@pytest.mark.parametrize(
    "environment,ref,sha",
    [
        ("test", "refs/heads/develop", "a" * 40),
        ("dev", "refs/heads/test/v1", "a" * 40),
        ("dev", "refs/heads/develop", "latest"),
    ],
)
def test_scope_refuses_before_commands(environment: str, ref: str, sha: str) -> None:
    with pytest.raises(ValueError):
        cutover.validate_scope(environment, ref, sha)


def test_fixed_exec_sources_vault_without_input_shell_code() -> None:
    driver = cutover.Driver("a" * 40)
    with patch.object(driver, "command", return_value="{}") as command:
        driver.pod_exec("probe", "backend", "inspect")
        args = command.call_args.args[0]
    assert args[-1] == "inspect"
    assert ". /vault/secrets/config;" in args[-3]
    assert 'exec python -m onyx.db.regulatory_annex_dev_cutover "$@"' in args[-3]


def test_workflow_rollback_applies_uniformly_to_every_environment() -> None:
    """DEV shares the same rollback-on-failure path as test/prod — the
    dev-only cutover-gated rollback was reverted as an unwanted deviation
    from the standard deployment flow."""
    import yaml

    workflow = yaml.safe_load(
        Path(
            ".github/workflows/customs-regulations-backend-lite-codebuild.yaml"
        ).read_text()
    )
    steps = workflow["jobs"]["build-and-deploy"]["steps"]
    rollback = next(step for step in steps if step["name"] == "Rollback on Failure")
    assert "env.env_x" not in rollback["if"]


def test_real_probe_refuses_aliases_and_lifecycle_without_mutation() -> None:
    from onyx.db.regulatory_annex_dev_cutover import inspect_indices

    client = Mock()
    name = "chunks_dev_exact"
    for metadata in (
        {"aliases": {"alias": {}}, "settings": {"index.uuid": "u"}},
        {
            "aliases": {},
            "settings": {"index.uuid": "u", "index.lifecycle.name": "rollover"},
        },
    ):
        client.indices.get.return_value = {name: metadata}
        with pytest.raises(RuntimeError):
            inspect_indices(client, [name])
    assert not client.indices.add_block.called


def test_shared_scope_never_lists_cluster_tasks() -> None:
    from onyx.db.regulatory_annex_dev_cutover import drain_server_work

    client = Mock()
    client.tasks.get.return_value = {"completed": True}
    drain_server_work(client, {"mode": "shared-scoped", "task_ids": ["devNode:12"]})
    client.tasks.get.assert_called_once_with(task_id="devNode:12")
    client.tasks.list.assert_not_called()
    client.cluster.pending_tasks.assert_not_called()


def test_dedicated_scope_waits_for_server_write_and_metadata_work() -> None:
    from onyx.db.regulatory_annex_dev_cutover import drain_server_work

    client = Mock()
    client.tasks.list.side_effect = [
        {"nodes": {"node": {"tasks": {"1": {}}}}},
        {"nodes": {}},
    ]
    client.cluster.pending_tasks.side_effect = [
        {"tasks": [{"source": "metadata"}]},
        {"tasks": []},
    ]
    with patch("onyx.db.regulatory_annex_dev_cutover.time.sleep") as sleep:
        drain_server_work(client, {"mode": "dev-dedicated"})
    assert client.tasks.list.call_count == 2
    sleep.assert_called_once_with(2)


def test_barrier_partial_ack_never_unblocks() -> None:
    from onyx.db import regulatory_annex_dev_cutover as probe

    name = "chunks_dev_exact"
    client = Mock()
    client.info.return_value = {"cluster_uuid": "cluster"}
    client.indices.get.return_value = {
        name: {"settings": {"index.uuid": "u"}, "aliases": {}}
    }
    client.tasks.list.return_value = {"nodes": {}}
    client.cluster.pending_tasks.return_value = {"tasks": []}
    client.indices.add_block.return_value = {
        "acknowledged": True,
        "shards_acknowledged": False,
    }
    wrapper = Mock()
    wrapper.__enter__ = Mock(return_value=wrapper)
    wrapper.__exit__ = Mock(return_value=False)
    wrapper.publication_client.return_value = client
    with (
        patch.object(probe, "configured_indices", return_value=[name]),
        patch(
            "onyx.document_index.elasticsearch.client.ElasticsearchClient",
            return_value=wrapper,
        ),
    ):
        with pytest.raises(RuntimeError, match="write_barrier_unacknowledged"):
            probe.operate(
                "block",
                {name: "u"},
                {
                    "mode": "dev-dedicated",
                    "cluster_uuid": "cluster",
                    "evidence_ref": "review",
                    "expires_at": 9999999999,
                    "indices": [name],
                },
            )
    client.indices.put_settings.assert_not_called()


def test_old_worker_drain_cancels_before_wait_and_never_purges() -> None:
    from scripts import regulatory_annex_dev_drain as drain

    events: list[str] = []
    workers = [name for name in drain.WORKERS if name != "regulatory_annex"]
    status = "\n".join(
        [f"celery_worker_{name} RUNNING pid 1, uptime 1:00" for name in workers]
        + [f"{name} RUNNING pid 2, uptime 1:00" for name in drain.BEATS]
    )
    app = Mock()

    def inspection(destination: list[str], timeout: int) -> Mock:
        assert timeout == 10
        inspector = Mock()
        inspector.stats.return_value = {name: {"pid": 1} for name in destination}
        inspector.active_queues.return_value = {
            name: [{"name": "old_queue"}] if len(destination) == 1 else []
            for name in destination
        }
        for field in ("active", "reserved", "scheduled"):
            getattr(inspector, field).return_value = {name: [] for name in destination}
        events.append("inspect")
        return inspector

    app.control.inspect.side_effect = inspection

    def cancel(
        queue: str, destination: list[str], reply: bool, timeout: int
    ) -> list[dict[str, dict[str, str]]]:
        assert queue == "old_queue" and reply and timeout == 10
        events.append("cancel")
        return [{destination[0]: {"ok": "cancelled"}}]

    app.control.cancel_consumer.side_effect = cancel

    def run(args: list[str]) -> str:
        events.append(args[-2] + ":" + args[-1])
        return status if args[-1] == "status" else ""

    with (
        patch.object(drain, "run", side_effect=run),
        patch.object(drain, "Celery", return_value=app),
        patch.object(drain.socket, "gethostname", return_value="pod"),
    ):
        drain.drain()
    last_cancel = max(i for i, event in enumerate(events) if event == "cancel")
    first_worker_stop = next(
        i for i, event in enumerate(events) if event.startswith("stop:celery_worker")
    )
    assert first_worker_stop > last_cancel
    app.control.purge.assert_not_called()


@pytest.mark.parametrize("env_x,expect_rollback", [("dev", True), ("test-v1", True)])
def test_actual_workflow_pending_release_branch(
    env_x: str, expect_rollback: bool, tmp_path: Path
) -> None:
    """DEV now recovers a pending-upgrade release the same way test/prod do —
    the dev-only refusal was reverted as an unwanted deviation from the
    standard deployment flow."""
    import os
    import subprocess

    import yaml

    workflow = yaml.safe_load(
        Path(
            ".github/workflows/customs-regulations-backend-lite-codebuild.yaml"
        ).read_text()
    )
    deploy = next(
        step["run"]
        for step in workflow["jobs"]["build-and-deploy"]["steps"]
        if step["name"] == "Deploy api and background with Helm"
    )
    function = deploy[: deploy.index("verify_benchmark_worker()")]
    events = tmp_path / "events"
    harness = (
        """
    kubectl() {
      echo "kubectl $*" >> "$EVENTS"
      if [[ "$*" == *"get pods"* ]]; then echo '{"items":[]}'; fi
    }
    helm() {
      echo "helm $*" >> "$EVENTS"
      case "$1" in
        status) echo '{"info":{"status":"pending-upgrade"}}' ;;
        history) echo '[{"status":"deployed","revision":1}]' ;;
        upgrade) return 1 ;;
      esac
    }
    """
        + function
        + '\ndeploy_app customs-regulations-api missing true ""\n'
    )
    result = subprocess.run(
        ["bash", "-e", "-c", harness],
        env={
            **os.environ,
            "env_x": env_x,
            "namespace": "customs-regulations-dev"
            if env_x == "dev"
            else "customs-regulations-test",
            "space_x": env_x,
            "IMAGE_TAG": "a" * 40,
            "EVENTS": str(events),
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    commands = events.read_text()
    assert ("helm rollback" in commands) is expect_rollback
    assert "--force" not in commands


def test_reappearing_autoscaler_refuses_before_unblock() -> None:
    driver = cutover.Driver("a" * 40)
    with (
        patch.object(
            driver, "get", return_value={"items": [{"metadata": {"name": "hpa"}}]}
        ),
        patch.object(driver, "pod_exec") as execute,
    ):
        with pytest.raises(RuntimeError, match="writer_autoscaler_appeared"):
            driver.assert_no_writers()
    execute.assert_not_called()


def test_existing_protected_release_does_not_require_another_first_cutover() -> None:
    driver = Mock(spec=cutover.Driver)
    driver.continue_protected_release.return_value = True
    cutover.prepare(driver)
    driver.record_installing.assert_called_once()
    driver.inventory.assert_not_called()
    driver.block_indices.assert_not_called()


def test_actual_prepare_commands_keep_old_pods_gone_before_unblock() -> None:
    import json

    driver = cutover.Driver("a" * 40)
    driver.container = "background"
    driver.indices = {"chunks_dev_exact": "uuid"}
    driver.scope = {"mode": "dev-dedicated"}
    events: list[list[str]] = []

    def command(args: list[str], stdin: str | None = None, timeout: int = 900) -> str:
        assert timeout == 900
        events.append(args)
        if "get" in args and "horizontalpodautoscalers" in args:
            return '{"items":[]}'
        if "get" in args and "pods" in args:
            return '{"items":[]}'
        if "get" in args and "deployment" in args:
            return '{"spec":{"replicas":0}}'
        if "apply" in args:
            assert stdin is not None
            assert json.loads(stdin)["data"]["phase"] in {"blocked", "installing"}
        if "exec" in args:
            assert args[-1] in {"block", "unblock"}
            assert json.loads(stdin or "{}")["indices"] == driver.indices
        return "{}"

    with patch.object(driver, "command", side_effect=command):
        driver.stop_api()
        driver.stop_background()
        driver.assert_no_writers()
        driver.block_indices()
        driver.record_blocked()
        driver.assert_no_writers()
        driver.unblock_indices()
        driver.record_installing()
    block = next(i for i, args in enumerate(events) if args[-1] == "block")
    unblock = next(i for i, args in enumerate(events) if args[-1] == "unblock")
    assert (
        len(
            [
                args
                for args in events[:block]
                if "wait" in args and "--for=delete" in args
            ]
        )
        == 2
    )
    assert any("horizontalpodautoscalers" in args for args in events[block:unblock])
    assert not any("--force" in args or "rollback" in args for args in events)


def test_activation_requires_both_workflows_before_provider_probe() -> None:
    driver = Mock(spec=cutover.Driver)
    driver.sha = "a" * 40
    driver.get.return_value = {"data": {"sha": driver.sha, "phase": "released"}}
    with (
        patch.object(
            cutover, "require_release_runs", side_effect=RuntimeError("web pending")
        ),
        patch.object(cutover, "acceptance") as acceptance,
        patch.object(cutover, "deploy_same_image") as deploy,
    ):
        with pytest.raises(RuntimeError):
            cutover.verify_or_activate(driver, True)
    acceptance.assert_not_called()
    deploy.assert_not_called()


@pytest.mark.parametrize("http_status", [200, 403])
def test_frontend_health_uses_release_identity_and_reports_http_status(
    http_status: int,
) -> None:
    driver = Mock(spec=cutover.Driver)
    driver.sha = "a" * 40
    driver.pods.return_value = [
        {
            "metadata": {},
            "spec": {
                "containers": [
                    {
                        "image": "255114580789.dkr.ecr.eu-central-1.amazonaws.com/"
                        "customs-regulations-web-dev:" + driver.sha
                    }
                ]
            },
        }
    ]
    response = Mock()
    response.__enter__ = Mock(return_value=Mock(status=200))
    response.__exit__ = Mock(return_value=False)

    def open_health(request: Request, *, timeout: int) -> Mock:
        assert isinstance(request, Request)
        assert request.full_url == (
            "https://dev-customs-regulations.singlewindow.io/api/health"
        )
        assert request.get_header("User-agent") == "Onyx-DEV-Release/1.0"
        assert timeout == 30
        if http_status != 200:
            raise HTTPError(
                request.full_url, http_status, "private-response", Message(), None
            )
        return response

    with patch.object(cutover.urllib.request, "urlopen", side_effect=open_health):
        if http_status == 200:
            cutover.verify_frontend(driver)
        else:
            with pytest.raises(
                cutover.CutoverRefusal, match="^frontend_API_health_HTTP_403$"
            ):
                cutover.verify_frontend(driver)


def test_failed_canary_disables_creation_on_the_same_binary() -> None:
    driver = Mock(spec=cutover.Driver)
    driver.sha = "a" * 40
    driver.get.return_value = {"data": {"sha": driver.sha, "phase": "released"}}
    with (
        patch.object(cutover, "require_release_runs"),
        patch.object(cutover, "verify_frontend"),
        patch.object(
            cutover, "acceptance", side_effect=[None, RuntimeError("canary failed")]
        ),
        patch.object(cutover, "deploy_same_image") as deploy,
    ):
        with pytest.raises(RuntimeError):
            cutover.verify_or_activate(driver, True)
    assert [call.args for call in deploy.call_args_list] == [
        (driver, True),
        (driver, False),
    ]


def test_reserved_or_scheduled_work_prevents_worker_stop() -> None:
    from scripts import regulatory_annex_dev_drain as drain

    destination = "light@pod"
    app = Mock()
    inspector = app.control.inspect.return_value
    inspector.stats.return_value = {destination: {"pid": 1}}
    inspector.active_queues.side_effect = [
        {destination: [{"name": "metadata"}]},
        {destination: []},
    ]
    inspector.active.return_value = {destination: []}
    inspector.reserved.return_value = {destination: []}
    inspector.scheduled.return_value = {destination: [{"id": "future-delivery"}]}
    app.control.cancel_consumer.return_value = [{destination: {"ok": "cancelled"}}]
    with (
        patch.object(drain, "WORKERS", ("light",)),
        patch.object(drain, "BEATS", ()),
        patch.object(
            drain, "run", return_value="celery_worker_light RUNNING pid 1, uptime 1:00"
        ) as run,
        patch.object(drain, "Celery", return_value=app),
        patch.object(drain.socket, "gethostname", return_value="pod"),
        patch.object(drain.time, "monotonic", side_effect=[0, 1, 601]),
        patch.object(drain.time, "sleep"),
    ):
        with pytest.raises(RuntimeError, match="drain_timeout_queues_preserved"):
            drain.drain()
    assert all("stop" not in call.args[0] for call in run.call_args_list)
    app.control.purge.assert_not_called()


def test_bootstrap_inventory_does_not_stop_writers_or_require_scope() -> None:
    driver = Mock(spec=cutover.Driver)
    driver.container = "backend"
    driver.pod_exec.return_value = (
        '{"cluster_uuid":"dev","indices":{"chunks_dev_exact":"uuid"}}'
    )
    cutover.inventory_only(driver)
    driver.inventory.assert_called_once_with(require_scope=False)
    driver.delete_probe.assert_called_once()
    driver.stop_api.assert_not_called()
    driver.block_indices.assert_not_called()


def test_unknown_scope_refuses_without_task_inventory() -> None:
    from onyx.db.regulatory_annex_dev_cutover import validate_scope_evidence

    client = Mock()
    with pytest.raises(
        RuntimeError, match="authoritative_DEV_ES_scope_evidence_required"
    ):
        validate_scope_evidence(client, {}, ["chunks_dev_exact"])
    client.tasks.list.assert_not_called()
    client.cluster.pending_tasks.assert_not_called()


def test_failed_acceptance_retains_only_bounded_probe_evidence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json
    import subprocess

    driver = cutover.Driver("a" * 40)
    report = {
        "phase": "preflight",
        "status": "failed",
        "release_sha_metadata": driver.sha,
        "calibration": {
            "status": "failed",
            "cases": [
                {
                    "format": "docx",
                    "supported": False,
                    "rationale": "Fixture mismatch",
                    "api_key": "secret",
                }
            ],
            "attempt_count": 1,
        },
        "raw_prompt": "private prompt",
    }
    result = subprocess.CompletedProcess(
        ["kubectl"],
        1,
        stdout="provider noise\n" + json.dumps(report),
        stderr="secret stderr",
    )
    with patch.object(cutover.subprocess, "run", return_value=result):
        with pytest.raises(cutover.CutoverRefusal):
            driver.command(["kubectl"], acceptance_phase="preflight")
    output = capsys.readouterr().out
    assert '"supported": false' in output
    assert "Fixture mismatch" in output
    assert "secret" not in output and "private prompt" not in output
    assert "provider noise" not in output


def test_acceptance_refuses_unbound_output(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(
        cutover.CutoverRefusal, match="fixed_acceptance_report_required"
    ):
        cutover.emit_acceptance_report(
            '{"phase":"preflight","status":"passed"}', "preflight", "a" * 40
        )
    assert not capsys.readouterr().out


def test_acceptance_retains_maximum_escaped_unicode_calibration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    rationale = "😀" * 4000
    report = {
        "phase": "preflight",
        "status": "passed",
        "release_sha_metadata": "a" * 40,
        "calibration": {
            "status": "passed",
            "cases": [
                {"rationale": rationale, "supported": value}
                for value in [True, False, True, False]
            ],
            "attempt_count": 4,
            "attempt_count_complete": True,
        },
    }
    serialized = json.dumps(report)
    assert len(serialized) > 100_000
    cutover.emit_acceptance_report(serialized, "preflight", "a" * 40)
    emitted = json.loads(capsys.readouterr().out)
    assert [case["rationale"] for case in emitted["calibration"]["cases"]] == [
        rationale
    ] * 4
    assert emitted["calibration"]["attempt_count_complete"] is True


@pytest.mark.parametrize("status", ["passed", "failed"])
def test_canary_report_retains_publication_and_cleanup_identities(
    status: str, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    evidence: dict[str, str | int | bool] = {
        key: value
        for value, key in enumerate(
            (
                "publication_generation",
                "canonical_changes",
                "context_consumers",
                "embeddings",
                "exact_vector_reuses",
                "historical_projections",
                "retired_projections",
                "total_projections",
            ),
            1,
        )
    }
    evidence["persona_cleanup_complete"] = status == "passed"
    evidence["persona_cleanup_failure"] = "safe failure receipt"
    evidence["cleanup_complete"] = status == "passed"
    canary = {
        "evidence": evidence,
        "retained_objects": [
            {
                "kind": "tombstone",
                "id": "owned-projection",
                "index_name": "physical-index",
                "index_uuid": "physical-uuid",
            },
            {"kind": "review", "id": "owned-review"},
        ],
        "vision_roles": [
            {
                "side": "new",
                "position": 3,
                "table_role": "column_header",
                "source_sha256": "b" * 64,
                "locator_sha256": "c" * 64,
            }
        ],
        "creation_intents": [
            {"kind": "markdown", "marker": "owned-marker", "artifact_id": None}
        ],
        "persona_id": 123,
        "api_key": "secret",
    }
    report = {
        "phase": "canary",
        "status": status,
        "release_sha_metadata": "a" * 40,
        "canary": canary,
    }
    if status == "failed":
        with pytest.raises(
            cutover.CutoverRefusal, match="fixed_acceptance_probe_failed"
        ):
            cutover.emit_acceptance_report(json.dumps(report), "canary", "a" * 40)
    else:
        cutover.emit_acceptance_report(json.dumps(report), "canary", "a" * 40)
    emitted = json.loads(capsys.readouterr().out)["canary"]
    assert emitted == {key: value for key, value in canary.items() if key != "api_key"}


def _observed_response(body: object, status: int = 200) -> object:
    from elastic_transport import (
        ApiResponseMeta,
        HttpHeaders,
        NodeConfig,
        ObjectApiResponse,
    )

    return ObjectApiResponse(
        body=body,
        meta=ApiResponseMeta(
            status=status,
            http_version="1.1",
            headers=HttpHeaders(),
            duration=0,
            node=NodeConfig("http", "localhost", 29200),
        ),
    )


def test_shared_observed_scope_requires_exact_identity_without_attestations() -> None:
    from onyx.db.regulatory_annex_dev_cutover import validate_scope_evidence

    client = Mock()
    client.info.return_value = {"cluster_uuid": "actual"}
    evidence = {
        "mode": "shared-observed",
        "cluster_uuid": "actual",
        "indices": ["chunks_dev_exact"],
        "expires_at": 9999999999,
        "evidence_ref": "observed release",
    }
    validate_scope_evidence(client, evidence, ["chunks_dev_exact"])
    for key, value in (
        ("cluster_uuid", "other"),
        ("indices", []),
        ("expires_at", 0),
        ("evidence_ref", ""),
    ):
        with pytest.raises(RuntimeError, match="scope_evidence_required"):
            validate_scope_evidence(
                client, {**evidence, key: value}, ["chunks_dev_exact"]
            )


@pytest.mark.parametrize(
    "endpoint,body,status,wrapped",
    [
        ("tasks", {}, 200, False),
        ("tasks", None, 200, True),
        ("tasks", [], 200, True),
        ("tasks", {}, 503, True),
        ("tasks", {"unexpected": []}, 200, True),
        ("tasks", {"tasks": {}}, 200, True),
        ("tasks", {"tasks": [{}]}, 200, True),
        ("tasks", {"tasks": [{"id": True, "node": "n"}]}, 200, True),
        (
            "tasks",
            {"tasks": [{"id": 1, "node": "n", "description": "hidden"}]},
            200,
            True,
        ),
        ("tasks", {"node_failures": [{}]}, 200, True),
        ("tasks", {"task_failures": [{"reason": "never log"}]}, 200, True),
        ("tasks", {"node_failures": None}, 200, True),
        ("tasks", {"error": {}}, 200, True),
        ("pending", {}, 200, False),
        ("pending", None, 200, True),
        ("pending", {"tasks": [None]}, 200, True),
        ("pending", {"tasks": [{"insert_order": -1}]}, 200, True),
        ("pending", {"tasks": [{"insert_order": 1, "source": "hidden"}]}, 200, True),
        ("pending", {"status": 500}, 200, True),
    ],
)
def test_shared_observed_refuses_incomplete_operational_responses(
    endpoint: str, body: object, status: int, wrapped: bool
) -> None:
    from onyx.db.regulatory_annex_dev_cutover import drain_server_work

    client = Mock()
    client.options.return_value = client
    client.tasks.list.return_value = _observed_response({})
    client.cluster.pending_tasks.return_value = _observed_response({})
    response = _observed_response(body, status) if wrapped else body
    if endpoint == "tasks":
        client.tasks.list.return_value = response
    else:
        client.cluster.pending_tasks.return_value = response
    with pytest.raises(RuntimeError, match="server_task_inventory_incomplete"):
        drain_server_work(client, {"mode": "shared-observed"})
    client.tasks.get.assert_not_called()


def test_shared_observed_waits_for_quiet_minimal_metadata() -> None:
    from onyx.db.regulatory_annex_dev_cutover import drain_server_work

    client = Mock()
    client.options.return_value = client
    client.tasks.list.side_effect = [
        _observed_response({"tasks": [{"id": 3, "node": "node"}]}),
        _observed_response({}),
        _observed_response({}),
    ]
    client.cluster.pending_tasks.side_effect = [
        _observed_response({}),
        _observed_response({"tasks": [{"insert_order": 1}]}),
        _observed_response({}),
    ]
    with patch("onyx.db.regulatory_annex_dev_cutover.time.sleep") as sleep:
        drain_server_work(client, {"mode": "shared-observed"})
    assert sleep.call_count == 2
    client.tasks.list.assert_called_with(
        actions=[
            "indices:data/write/*",
            "indices:admin/*",
            "cluster:admin/snapshot/restore*",
        ],
        detailed=False,
        group_by="none",
        timeout="10s",
        filter_path=[
            "tasks.node",
            "tasks.id",
            "node_failures",
            "task_failures",
            "error",
            "status",
        ],
    )
    client.cluster.pending_tasks.assert_called_with(
        filter_path=["tasks.insert_order", "error", "status"], master_timeout="10s"
    )
    client.options.assert_called_with(request_timeout=15)
    client.tasks.get.assert_not_called()
    client.tasks.cancel.assert_not_called()


def test_shared_observed_active_work_has_bounded_wait() -> None:
    from onyx.db import regulatory_annex_dev_cutover as probe

    client = Mock()
    client.options.return_value = client
    client.tasks.list.return_value = _observed_response(
        {"tasks": [{"id": 3, "node": "node"}]}
    )
    client.cluster.pending_tasks.return_value = _observed_response({})
    with (
        patch.object(probe.time, "monotonic", side_effect=[0, 1, 601]),
        patch.object(probe.time, "sleep"),
    ):
        with pytest.raises(RuntimeError, match="server_work_not_drained"):
            probe.drain_server_work(client, {"mode": "shared-observed"})


def test_shared_observed_rechecks_quiet_before_block_and_unblock() -> None:
    from onyx.db import regulatory_annex_dev_cutover as probe

    name = "chunks_dev_exact"
    client = Mock()
    client.options.return_value = client
    client.info.return_value = {"cluster_uuid": "actual"}
    client.indices.get.return_value = {
        name: {"settings": {"index.uuid": "u"}, "aliases": {}}
    }
    client.tasks.list.side_effect = [
        _observed_response({}),
        _observed_response({"node_failures": [{}]}),
    ]
    client.cluster.pending_tasks.return_value = _observed_response({})
    client.indices.add_block.return_value = {
        "acknowledged": True,
        "shards_acknowledged": True,
        "indices": [{"name": name, "blocked": True}],
    }
    wrapper = Mock()
    wrapper.__enter__ = Mock(return_value=wrapper)
    wrapper.__exit__ = Mock(return_value=False)
    wrapper.publication_client.return_value = client
    scope = {
        "mode": "shared-observed",
        "cluster_uuid": "actual",
        "indices": [name],
        "expires_at": 9999999999,
        "evidence_ref": "release",
    }
    with (
        patch.object(probe, "configured_indices", return_value=[name]),
        patch(
            "onyx.document_index.elasticsearch.client.ElasticsearchClient",
            return_value=wrapper,
        ),
    ):
        assert probe.operate("block", {name: "u"}, scope) == {name: "u"}
        with pytest.raises(RuntimeError, match="server_task_inventory_incomplete"):
            probe.operate("unblock", {name: "u"}, scope)
    client.indices.add_block.assert_called_once()
    client.indices.put_settings.assert_not_called()


def test_diagnose_alone_can_inspect_an_earlier_exact_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, runner = "a" * 40, "b" * 40
    for key, value in {
        "env_x": "dev",
        "GITHUB_REF": "refs/heads/develop",
        "IMAGE_TAG": runtime,
        "GITHUB_SHA": runner,
    }.items():
        monkeypatch.setenv(key, value)
    driver = Mock(spec=cutover.Driver)
    with (
        patch.object(cutover, "Driver", return_value=driver),
        patch.object(cutover, "diagnose_release", create=True) as diagnose,
    ):
        monkeypatch.setattr("sys.argv", ["cutover", "annex-diagnose"])
        cutover.main()
        diagnose.assert_called_once_with(driver, runner)
        for phase in (
            "prepare",
            "release",
            "failure",
            "values",
            "verify",
            "annex-inventory",
        ):
            monkeypatch.setattr("sys.argv", ["cutover", phase])
            with pytest.raises(RuntimeError, match="checked_out"):
                cutover.main()


def test_diagnose_checks_released_runtime_and_executes_only_fixed_wrapper(
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = Mock(spec=cutover.Driver)
    driver.sha = "a" * 40
    driver.get.return_value = {"data": {"sha": driver.sha, "phase": "released"}}
    driver.pods.return_value = [
        {
            "metadata": {"name": "background"},
            "spec": {
                "containers": [
                    {"name": "worker", "image": cutover.REPOSITORY + ":" + driver.sha}
                ]
            },
        }
    ]
    driver.command.return_value = '{"stages":[{"stage":"configuration","exception_class":"UnicodeDecodeError","frames":[{"filename":"onyx/db/config.py","function":"read","line":7}]},{"stage":"native_parser","exception_class":null,"frames":[]}]}'
    with (
        patch.object(cutover, "require_release_runs") as runs,
        patch.object(cutover, "verify_frontend") as frontend,
    ):
        cutover.diagnose_release(driver, "b" * 40)
    runs.assert_called_once_with(driver.sha)
    driver.verify_runtime.assert_called_once_with(
        readiness=False, redis_isolation=False
    )
    frontend.assert_called_once_with(driver)
    call = driver.command.call_args
    assert call.kwargs["timeout"] == 180
    command = call.args[0]
    assert command[:4] == ["kubectl", "--namespace", cutover.NAMESPACE, "exec"]
    assert command[-1] == cutover.DIAGNOSTIC_PROGRAM
    assert "UnicodeDecodeError" in capsys.readouterr().out
    driver.get.return_value["data"]["sha"] = "c" * 40
    with pytest.raises(RuntimeError, match="matching_cutover"):
        cutover.diagnose_release(driver, "b" * 40)


def test_fixed_diagnostic_preserves_uninitialized_failure_and_omits_messages(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys
    from types import ModuleType

    db = ModuleType("onyx.db.regulatory_annex_acceptance")
    probe = ModuleType("onyx.regulatory.amendments.annexes.dev_acceptance")
    namespace: dict[str, object] = {}
    exec(
        compile(
            'def verify_dev_configuration():\n    raise UnicodeDecodeError("utf8", b"secret", 0, 1, "PASSWORD=secret")',
            "/app/onyx/db/fixture.py",
            "exec",
        ),
        namespace,
    )
    setattr(db, "verify_dev_configuration", namespace["verify_dev_configuration"])
    setattr(probe, "validate_scope", lambda **_: None)
    setattr(probe, "native_parser_probe", lambda: {"untrusted": "DO_NOT_PRINT"})
    monkeypatch.setitem(sys.modules, db.__name__, db)
    monkeypatch.setitem(sys.modules, probe.__name__, probe)
    exec(cutover.DIAGNOSTIC_PROGRAM, {"__name__": "__main__"})
    output = capsys.readouterr().out
    assert "UnicodeDecodeError" in output and "onyx/db/fixture.py" in output
    assert "PASSWORD" not in output and "DO_NOT_PRINT" not in output
    assert "secret" not in output and "/app/" not in output
    assert "set_is_ee" not in cutover.DIAGNOSTIC_PROGRAM


def test_failed_acceptance_stage_diagnostic_survives_safe_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    report = {
        "phase": "preflight",
        "status": "failed",
        "release_sha_metadata": "a" * 40,
        "failure_stage": "configuration",
        "exception_type": "UnicodeDecodeError",
        "failure": json.dumps(
            {
                "stage": "configuration",
                "exceptions": [
                    {
                        "type": "UnicodeDecodeError",
                        "frames": [
                            {
                                "module": "onyx.utils.encryption",
                                "function": "_decrypt_bytes",
                                "line": 34,
                            }
                        ],
                    }
                ],
            }
        ),
        "exception_message": "DO_NOT_LOG_source_or_secret",
    }
    with pytest.raises(cutover.CutoverRefusal, match="fixed_acceptance_probe_failed"):
        cutover.emit_acceptance_report(json.dumps(report), "preflight", "a" * 40)
    output = capsys.readouterr().out
    retained = json.loads(output)
    assert retained["failure_stage"] == "configuration"
    assert retained["exception_type"] == "UnicodeDecodeError"
    assert retained["failure"] == report["failure"]
    assert "DO_NOT_LOG" not in output


@pytest.mark.parametrize("key", ["failure_stage", "exception_type"])
def test_acceptance_stage_diagnostics_refuse_arbitrary_values(key: str) -> None:
    import json

    report = {
        "phase": "preflight",
        "status": "failed",
        "release_sha_metadata": "a" * 40,
        key: "DO_NOT_LOG_source_or_secret",
    }
    with pytest.raises(cutover.CutoverRefusal, match="acceptance_.*_refused"):
        cutover.emit_acceptance_report(json.dumps(report), "preflight", "a" * 40)


def runtime_pod(
    *,
    deleting: bool = False,
    ready: bool = True,
    sha: str = "a" * 40,
    restarts: int = 0,
) -> dict[str, Any]:
    return {
        "metadata": {
            "name": "private-pod",
            **({"deletionTimestamp": "now"} if deleting else {}),
        },
        "spec": {
            "containers": [{"name": "main", "image": f"{cutover.REPOSITORY}:{sha}"}]
        },
        "status": {
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
            "containerStatuses": [{"restartCount": restarts}],
        },
    }


def test_runtime_waits_for_all_old_pods_then_checks_worker() -> None:
    driver = cutover.Driver("a" * 40)
    with (
        patch.object(
            driver,
            "pods",
            side_effect=[
                [runtime_pod(deleting=True), runtime_pod()],
                [],
                [runtime_pod(ready=False)],
                [runtime_pod()],
                [runtime_pod()],
            ],
        ),
        patch.object(driver, "worker_readiness_probe", return_value=True) as command,
        patch.object(driver, "redis_isolation"),
        patch("time.sleep") as sleep,
    ):
        driver.verify_runtime()
    assert sleep.call_count == 3
    assert command.call_count == 1
    assert command.call_args.kwargs["timeout"] <= 60


@pytest.mark.parametrize(
    "pod,reason",
    [
        (runtime_pod(deleting=True, sha="b" * 40), "compatible_exact_SHA_required"),
        (runtime_pod(restarts=1), "new_pod_restarted"),
    ],
)
def test_runtime_never_waits_away_wrong_images_or_restarts(
    pod: dict[str, Any], reason: str
) -> None:
    driver = cutover.Driver("a" * 40)
    with (
        patch.object(driver, "pods", return_value=[pod]),
        patch("time.sleep") as sleep,
        pytest.raises(cutover.CutoverRefusal, match=reason),
    ):
        driver.verify_runtime()
    sleep.assert_not_called()


def test_runtime_timeout_reports_only_safe_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = cutover.Driver("a" * 40)
    with (
        patch.object(
            driver, "pods", return_value=[runtime_pod(deleting=True), runtime_pod()]
        ),
        patch("time.monotonic", side_effect=[0, 181]),
        patch("time.sleep") as sleep,
        pytest.raises(
            cutover.CutoverRefusal, match="one_ready_compatible_pod_required"
        ),
    ):
        driver.verify_runtime()
    sleep.assert_not_called()
    output = capsys.readouterr().out
    assert (
        '"pods": 2' in output
        and '"deleting": 1' in output
        and '"ready": 2' in output
        and '"wrong_sha": 0' in output
    )
    assert "private-pod" not in output


def runner_comparison() -> dict[str, Any]:
    return {
        "status": "ahead",
        "ahead_by": 1,
        "behind_by": 0,
        "total_commits": 1,
        "base_commit": {"sha": "a" * 40},
        "merge_base_commit": {"sha": "a" * 40},
        "commits": [{"sha": "b" * 40}],
        "files": [
            {
                "filename": "backend/scripts/regulatory_annex_dev_cutover.py",
                "status": "modified",
            }
        ],
    }


def comparison_response(
    body: dict[str, Any], *, status: int = 200, link: str = ""
) -> Mock:
    import json

    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.status = status
    response.headers = {"Link": link}
    response.read.return_value = json.dumps(body).encode()
    return response


def test_runner_comparison_authenticates_exact_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "atezsoftware/customs-regulations-chatbot")
    monkeypatch.setenv("GH_TOKEN", "test-only")
    with patch.object(
        cutover.urllib.request,
        "urlopen",
        return_value=comparison_response(runner_comparison()),
    ) as request:
        cutover.require_runner_compatibility("a" * 40, "b" * 40)
    args = request.call_args.args[0]
    assert args.full_url.endswith("/compare/" + "a" * 40 + "..." + "b" * 40)
    assert args.get_header("Authorization") == "Bearer test-only"


@pytest.mark.parametrize(
    "change",
    [
        {"status": "diverged"},
        {"behind_by": 1},
        {"base_commit": {"sha": "c" * 40}},
        {"merge_base_commit": {"sha": "c" * 40}},
        {"commits": []},
        {"commits": [{"sha": "c" * 40}]},
        {"total_commits": 251},
        {"files": []},
        {"files": [{"filename": "backend/onyx/main.py", "status": "modified"}]},
        {
            "files": [
                {
                    "filename": "backend/scripts/regulatory_annex_dev_cutover.py",
                    "status": "renamed",
                    "previous_filename": "other",
                }
            ]
        },
        {
            "files": [
                {
                    "filename": "backend/scripts/regulatory_annex_dev_cutover.py",
                    "status": "modified",
                    "previous_filename": "other",
                }
            ]
        },
        {"truncated": True},
        {
            "files": [
                {
                    "filename": "backend/scripts/regulatory_annex_dev_cutover.py",
                    "status": "modified",
                }
            ]
            * 300
        },
    ],
)
def test_runner_comparison_refuses_unproven_delta(
    monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "atezsoftware/customs-regulations-chatbot")
    monkeypatch.setenv("GH_TOKEN", "test-only")
    with (
        patch.object(
            cutover.urllib.request,
            "urlopen",
            return_value=comparison_response(runner_comparison() | change),
        ),
        pytest.raises(cutover.CutoverRefusal, match="runner_only_comparison_required"),
    ):
        cutover.require_runner_compatibility("a" * 40, "b" * 40)


@pytest.mark.parametrize("status,link", [(206, ""), (200, '<next>; rel="next"')])
def test_runner_comparison_refuses_incomplete_http(
    monkeypatch: pytest.MonkeyPatch, status: int, link: str
) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "atezsoftware/customs-regulations-chatbot")
    monkeypatch.setenv("GH_TOKEN", "test-only")
    with (
        patch.object(
            cutover.urllib.request,
            "urlopen",
            return_value=comparison_response(
                runner_comparison(), status=status, link=link
            ),
        ),
        pytest.raises(cutover.CutoverRefusal, match="runner_only_comparison_required"),
    ):
        cutover.require_runner_compatibility("a" * 40, "b" * 40)


@pytest.mark.parametrize("phase", ["annex-verify", "annex-activate"])
def test_only_proven_runner_delta_can_verify_or_activate(
    phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in {
        "env_x": "dev",
        "GITHUB_REF": "refs/heads/develop",
        "IMAGE_TAG": "a" * 40,
        "GITHUB_SHA": "b" * 40,
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("sys.argv", ["cutover", phase])
    with (
        patch.object(cutover, "require_runner_compatibility") as compare,
        patch.object(cutover, "verify_or_activate") as verify,
    ):
        cutover.main()
        compare.assert_called_once_with("a" * 40, "b" * 40)
        assert verify.call_args.args[0].sha == "a" * 40
        assert verify.call_args.args[1] == (phase == "annex-activate")
        verify.reset_mock()
        compare.side_effect = cutover.CutoverRefusal("unproven")
        with pytest.raises(cutover.CutoverRefusal, match="unproven"):
            cutover.main()
        verify.assert_not_called()


@pytest.mark.parametrize(
    "body",
    [
        None,
        [],
        {},
        {"commits": None},
        {"commits": [], "files": None, "total_commits": 0},
    ],
)
def test_runner_comparison_refuses_malformed_body(
    body: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "atezsoftware/customs-regulations-chatbot")
    monkeypatch.setenv("GH_TOKEN", "test-only")
    with (
        patch.object(
            cutover.urllib.request, "urlopen", return_value=comparison_response(body)
        ),
        pytest.raises(cutover.CutoverRefusal, match="runner_only_comparison_required"),
    ):
        cutover.require_runner_compatibility("a" * 40, "b" * 40)


def test_runner_comparison_requires_authentication_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "atezsoftware/customs-regulations-chatbot")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    with (
        patch.object(cutover.urllib.request, "urlopen") as request,
        pytest.raises(
            cutover.CutoverRefusal, match="authenticated_exact_repository_required"
        ),
    ):
        cutover.require_runner_compatibility("a" * 40, "b" * 40)
    request.assert_not_called()


def test_worker_waits_for_fixed_readiness_and_rechecks_pods(
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = cutover.Driver("a" * 40)
    pending = cutover.subprocess.CompletedProcess(
        [],
        1,
        "private startup log\nNOT_READY annex_worker ValueError\n",
        "secret stderr",
    )
    ready = cutover.subprocess.CompletedProcess(
        [],
        0,
        "READY annex_worker scoped_queues registered_handlers concurrency_one\n",
        "",
    )
    with (
        patch.object(driver, "pods", return_value=[runtime_pod()]) as pods,
        patch.object(cutover.subprocess, "run", side_effect=[pending, ready]) as run,
        patch("time.sleep") as sleep,
    ):
        driver.verify_app("background")
    assert pods.call_count == 2 and sleep.call_count == 1
    assert run.call_args.kwargs["timeout"] <= 60
    output = capsys.readouterr().out
    assert (
        "NOT_READY annex_worker ValueError" in output and "READY annex_worker" in output
    )
    assert "private" not in output and "secret" not in output


@pytest.mark.parametrize(
    "code,stdout",
    [
        (0, ""),
        (1, ""),
        (0, "NOT_READY annex_worker ValueError"),
        (1, "READY annex_worker scoped_queues registered_handlers concurrency_one"),
        (0, "READY annex_worker wrong"),
        (1, "NOT_READY annex_worker ValueError secret"),
    ],
)
def test_worker_rejects_missing_or_invalid_fixed_report(
    code: int, stdout: str, capsys: pytest.CaptureFixture[str]
) -> None:
    driver = cutover.Driver("a" * 40)
    with (
        patch.object(driver, "pods", return_value=[runtime_pod()]),
        patch.object(
            cutover.subprocess,
            "run",
            return_value=cutover.subprocess.CompletedProcess(
                [], code, stdout, "secret"
            ),
        ),
        pytest.raises(cutover.CutoverRefusal, match="worker_readiness_report_required"),
    ):
        driver.verify_app("background")
    assert "secret" not in capsys.readouterr().out


def test_worker_never_ignores_restart_while_waiting() -> None:
    driver = cutover.Driver("a" * 40)
    with (
        patch.object(
            driver, "pods", side_effect=[[runtime_pod()], [runtime_pod(restarts=1)]]
        ),
        patch.object(
            cutover.subprocess,
            "run",
            return_value=cutover.subprocess.CompletedProcess(
                [], 1, "NOT_READY annex_worker ValueError", ""
            ),
        ) as run,
        patch("time.sleep"),
        pytest.raises(cutover.CutoverRefusal, match="new_pod_restarted"),
    ):
        driver.verify_app("background")
    assert run.call_count == 1


def test_worker_timeout_is_bounded_and_safe(capsys: pytest.CaptureFixture[str]) -> None:
    driver = cutover.Driver("a" * 40)
    with (
        patch.object(driver, "pods", return_value=[runtime_pod()]),
        patch.object(
            cutover.subprocess,
            "run",
            side_effect=cutover.subprocess.TimeoutExpired("secret", 60),
        ),
        patch("time.monotonic", side_effect=[0, 1, 601]),
        patch("time.sleep") as sleep,
        pytest.raises(cutover.CutoverRefusal, match="worker_readiness_timeout"),
    ):
        driver.verify_app("background")
    sleep.assert_not_called()
    assert "secret" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "enabled,order", [(True, ["background", "api"]), (False, ["api", "background"])]
)
def test_activation_consumer_ready_before_api_creation(
    enabled: bool, order: list[str]
) -> None:
    driver = Mock(spec=cutover.Driver)
    driver.sha = "a" * 40
    with patch.object(cutover, "render_values"):
        cutover.deploy_same_image(driver, enabled)
    events = [
        call
        for call in driver.method_calls
        if call[0] in {"command", "verify_app", "redis_isolation"}
    ]
    assert events[0].args[0][3] == f"customs-regulations-{order[0]}-dev"
    if enabled:
        assert events[1][0] == "verify_app" and events[1].args == ("background",)
        assert events[2][0] == "redis_isolation" and events[2].args == ("worker",)
        assert events[3].args[0][3] == "customs-regulations-api-dev"
    else:
        assert events[1].args[0][3] == "customs-regulations-background-dev"


@pytest.mark.parametrize("failure_method", ["verify_app", "redis_isolation"])
def test_failed_consumer_admission_never_enables_api(failure_method: str) -> None:
    driver = Mock(spec=cutover.Driver)
    driver.sha = "a" * 40
    getattr(driver, failure_method).side_effect = cutover.CutoverRefusal(
        "worker_readiness_timeout"
    )
    with patch.object(cutover, "render_values"), pytest.raises(cutover.CutoverRefusal):
        cutover.deploy_same_image(driver, True)
    assert driver.command.call_count == 1
    assert "background" in driver.command.call_args.args[0][3]


@pytest.mark.parametrize("operation", ["exec", "get", "rollout"])
@pytest.mark.parametrize("transport", [False, True])
def test_command_failure_exposes_only_safe_operation(
    operation: str, transport: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    driver = cutover.Driver("a" * 40)
    result = cutover.subprocess.CompletedProcess(
        [], 1, "secret stdout", "secret stderr"
    )
    with (
        patch.object(
            cutover.subprocess,
            "run",
            side_effect=OSError("secret") if transport else None,
            return_value=result,
        ),
        pytest.raises(cutover.CutoverRefusal, match="kubectl:" + operation),
    ):
        driver.command(
            ["kubectl", "--namespace", cutover.NAMESPACE, operation, "private"]
        )
    output = capsys.readouterr().out
    assert (
        "kubectl:" + operation in output
        and "secret" not in output
        and "private" not in output
    )


def test_worker_exec_transport_failure_has_safe_diagnostic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = cutover.Driver("a" * 40)
    with (
        patch.object(driver, "pods", return_value=[runtime_pod()]),
        patch.object(cutover.subprocess, "run", side_effect=OSError("secret")),
        pytest.raises(cutover.CutoverRefusal, match="kubectl:exec"),
    ):
        driver.verify_app("background")
    assert capsys.readouterr().out == "NOT_READY annex_worker ExecTransportFailure\n"


def cloudwatch_event(log: str, *, namespace: str = cutover.NAMESPACE) -> dict[str, Any]:
    import json

    return {
        "timestamp": 1789123650000,
        "logStreamName": "private",
        "message": json.dumps(
            {
                "kubernetes": {
                    "namespace_name": namespace,
                    "pod_name": "dev-customs-regulations-background-abc",
                    "container_name": "background",
                },
                "log": log,
            }
        ),
    }


def test_batch44_cloudwatch_trace_is_scoped_and_sanitized(
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace = 'Amendment batch 44 failed\nTraceback (most recent call last):\n  File "/app/onyx/regulatory/tasks.py", line 235, in run\n    secret_source()\nValueError: secret provider content\n'
    with patch.object(
        cutover,
        "cloudwatch_metadata",
        create=True,
        side_effect=[
            {
                "logGroups": [
                    {
                        "logGroupName": "/aws/containerinsights/atez-dev-cluster/application"
                    }
                ]
            },
            {"events": [cloudwatch_event(trace)]},
        ],
    ) as query:
        cutover.diagnose_batch44_logs("4a38658efa5014dfa2b039887bc7db0e12e8153e")
    output = capsys.readouterr().out
    assert '"exception_class": "ValueError"' in output
    assert '"filename": "onyx/regulatory/tasks.py"' in output
    assert "secret" not in output and "private" not in output
    args = query.call_args.args[0]
    assert args[0] == "filter-log-events"
    assert "kubernetes.namespace_name" in args[args.index("--filter-pattern") + 1]
    assert cutover.NAMESPACE in args[args.index("--filter-pattern") + 1]


def test_batch44_cloudwatch_absent_group_never_reads_content(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch.object(
        cutover, "cloudwatch_metadata", create=True, return_value={"logGroups": []}
    ) as query:
        cutover.diagnose_batch44_logs("4a38658efa5014dfa2b039887bc7db0e12e8153e")
    assert query.call_count == 1
    assert "application_group_absent" in capsys.readouterr().out


@pytest.mark.parametrize(
    "response,status",
    [
        (
            {
                "events": [
                    cloudwatch_event("secret", namespace="customs-regulations-test")
                ]
            },
            "namespace_scope_mismatch",
        ),
        ({"events": []}, "batch_trace_unavailable"),
        ({}, "invalid_response"),
        ({"events": [], "nextToken": "same"}, "pagination_incomplete"),
    ],
)
def test_batch44_cloudwatch_refuses_unproven_trace(
    response: dict[str, Any], status: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with patch.object(
        cutover,
        "cloudwatch_metadata",
        create=True,
        side_effect=[
            {
                "logGroups": [
                    {
                        "logGroupName": "/aws/containerinsights/atez-dev-cluster/application"
                    }
                ]
            },
            response,
            response,
        ],
    ):
        cutover.diagnose_batch44_logs("4a38658efa5014dfa2b039887bc7db0e12e8153e")
    output = capsys.readouterr().out
    assert status in output and "secret" not in output


def test_batch44_cloudwatch_other_runtime_does_not_query() -> None:
    with patch.object(cutover, "cloudwatch_metadata", create=True) as query:
        cutover.diagnose_batch44_logs("a" * 40)
    query.assert_not_called()


@pytest.mark.parametrize(
    "stderr,status",
    [
        ("AccessDeniedException secret", "access_denied"),
        ("other secret", "aws_query_failed"),
    ],
)
def test_batch44_cloudwatch_permission_errors_are_fixed(
    stderr: str, status: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with patch.object(
        cutover.subprocess,
        "run",
        return_value=cutover.subprocess.CompletedProcess([], 1, "secret", stderr),
    ) as command:
        cutover.diagnose_batch44_logs("4a38658efa5014dfa2b039887bc7db0e12e8153e")
    output = capsys.readouterr().out
    assert status in output and "secret" not in output
    assert command.call_args.args[0][:3] == ["aws", "logs", "describe-log-groups"]


def test_batch44_cloudwatch_requires_complete_pagination_before_trace(
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace = 'Amendment batch 44 failed\nTraceback (most recent call last):\n  File "/app/onyx/task.py", line 3, in run\nValueError: secret'
    with patch.object(
        cutover,
        "cloudwatch_metadata",
        side_effect=[
            {"logGroups": [{"logGroupName": cutover.BATCH44_GROUP}]},
            {"events": [cloudwatch_event(trace)], "nextToken": "next"},
            {"events": []},
        ],
    ) as query:
        cutover.diagnose_batch44_logs(cutover.BATCH44_RUNTIME)
    assert query.call_count == 3
    assert query.call_args.args[0][-2:] == ["--next-token", "next"]
    assert "trace_found" in capsys.readouterr().out


def test_logging_metadata_emits_only_collectors_and_safe_destinations(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    driver = Mock(spec=cutover.Driver)
    daemonsets = {
        "items": [
            {
                "metadata": {"name": "fluent-bit"},
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "fluent-bit",
                                    "image": "public.ecr.aws/aws-observability/aws-for-fluent-bit:stable",
                                    "env": [{"name": "SECRET", "value": "private"}],
                                }
                            ]
                        }
                    }
                },
            }
        ]
    }
    config = {
        "data": {
            "output.conf": "[OUTPUT]\n Name cloudwatch_logs\n log_group_name /custom/atez-dev/application\n region eu-central-1\n token private\n"
        }
    }
    driver.command.side_effect = [
        json.dumps(daemonsets),
        json.dumps({"items": []}),
        json.dumps({"items": []}),
        json.dumps({"items": []}),
        json.dumps(config),
    ]
    with patch.object(
        cutover,
        "cloudwatch_metadata",
        return_value={"logGroups": [{"logGroupName": "/custom/atez-dev/application"}]},
    ):
        cutover.diagnose_logging_metadata(driver)
    output = capsys.readouterr().out
    assert "fluent-bit" in output and "/custom/atez-dev/application" in output
    assert (
        "private" not in output
        and "SECRET" not in output
        and "output.conf" not in output
    )
    assert driver.command.call_count == 5
    assert all(
        call.args[0][1:4] == ["get", "daemonsets", "--namespace"]
        for call in driver.command.call_args_list[:4]
    )
    assert driver.command.call_args.args[0][1:4] == ["get", "configmap", "aws-logging"]


def test_logging_metadata_unavailable_is_precise_and_never_reads_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = Mock(spec=cutover.Driver)
    driver.command.side_effect = cutover.CutoverRefusal("command_failed:kubectl:get")
    with patch.object(
        cutover,
        "cloudwatch_metadata",
        return_value={"logGroups": [], "nextToken": "private"},
    ) as query:
        cutover.diagnose_logging_metadata(driver)
    output = capsys.readouterr().out
    assert (
        "unavailable" in output
        and '"complete": false' in output
        and "private" not in output
    )
    assert all(
        call.args[0][0] == "describe-log-groups" for call in query.call_args_list
    )


def test_fluentd_destination_follows_only_exact_referenced_configmaps(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    driver = Mock(spec=cutover.Driver)
    daemonset = {
        "items": [
            {
                "metadata": {"name": "fluentd"},
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "fluentd",
                                    "image": "fluent/fluentd-kubernetes-daemonset:v1.16.2-debian-elasticsearch7-1.0",
                                    "env": [
                                        {
                                            "name": "FLUENT_ELASTICSEARCH_HOST",
                                            "value": "elasticsearch.logging.svc",
                                        },
                                        {
                                            "name": "FLUENT_ELASTICSEARCH_PASSWORD",
                                            "value": "secret",
                                        },
                                    ],
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "config",
                                    "configMap": {"name": "fluentd-config"},
                                },
                                {
                                    "name": "secret",
                                    "secret": {"secretName": "never-read"},
                                },
                            ],
                        }
                    }
                },
            }
        ]
    }
    configs = {
        "metadata": {"name": "fluentd-config"},
        "data": {
            "fluent.conf": "<match **>\n @type elasticsearch\n host elasticsearch.logging.svc\n port 9200\n scheme http\n logstash_prefix fluentd\n password secret\n</match>"
        },
    }
    services = {
        "items": [
            {
                "metadata": {"name": "elasticsearch"},
                "spec": {"clusterIP": "10.0.0.1", "ports": [{"port": 9200}]},
            },
            {
                "metadata": {"name": "kibana"},
                "spec": {"clusterIP": "10.0.0.2", "ports": [{"port": 5601}]},
            },
        ]
    }
    values = [
        {"items": []},
        {"items": []},
        daemonset,
        {"items": []},
        configs,
        services,
        {"data": {}},
    ]
    driver.command.side_effect = [json.dumps(value) for value in values]
    with patch.object(cutover, "cloudwatch_metadata", return_value={"logGroups": []}):
        cutover.diagnose_logging_metadata(driver)
    output = capsys.readouterr().out
    assert (
        "elasticsearch.logging.svc" in output
        and "9200" in output
        and "kibana" in output
    )
    assert (
        "secret" not in output
        and "password" not in output
        and "never-read" not in output
    )
    assert driver.command.call_args_list[4].args[0][1:6] == [
        "get",
        "configmap",
        "fluentd-config",
        "--namespace",
        "logging",
    ]
    assert driver.command.call_args_list[5].args[0][1:5] == [
        "get",
        "services",
        "--namespace",
        "logging",
    ]
    assert all("secrets" not in call.args[0] for call in driver.command.call_args_list)


def test_fluentd_unknown_or_excessive_refs_never_trigger_dynamic_reads(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    driver = Mock(spec=cutover.Driver)
    daemonset = {
        "items": [
            {
                "metadata": {"name": "fluentd"},
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {"name": "fluentd", "image": "fluent/fluentd:1"}
                            ],
                            "volumes": [{"configMap": {"name": "../unsafe"}}],
                        }
                    }
                },
            }
        ]
    }
    driver.command.side_effect = [
        json.dumps(value)
        for value in [
            {"items": []},
            {"items": []},
            daemonset,
            {"items": []},
            {"data": {}},
        ]
    ]
    with patch.object(cutover, "cloudwatch_metadata", return_value={"logGroups": []}):
        cutover.diagnose_logging_metadata(driver)
    assert driver.command.call_count == 5
    output = capsys.readouterr().out
    assert "invalid_references" in output and "../unsafe" not in output


def es_trace_response() -> dict[str, Any]:
    lines = [
        "Amendment batch 44 failed",
        "Traceback (most recent call last):",
        '  File "/app/onyx/regulatory/tasks.py", line 235, in run',
        "    secret_source()",
        "ValueError: private provider content",
    ]
    return {
        "timed_out": False,
        "_shards": {"total": 2, "successful": 2, "failed": 0},
        "hits": {
            "total": {"value": len(lines), "relation": "eq"},
            "hits": [
                {
                    "_source": {
                        "@timestamp": "2026-09-11T10:47:30Z",
                        "kubernetes": {
                            "namespace_name": cutover.NAMESPACE,
                            "pod_name": "dev-customs-regulations-background-a",
                        },
                        "log": line,
                    }
                }
                for line in lines
            ],
        },
    }


def execute_es_diagnostic(
    data: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
    *,
    env: dict[str, str] | None = None,
) -> tuple[dict[str, Any], Mock]:
    import inspect
    import json

    response = Mock()
    response.status = 200
    response.read.return_value = json.dumps(data).encode()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    opener = Mock()
    opener.open.return_value = response
    with (
        patch.dict("os.environ", env or {}, clear=True),
        patch("urllib.request.build_opener", return_value=opener),
    ):
        exec(
            cutover.BATCH44_ES_PROGRAM
            + "\n"
            + inspect.getsource(cutover.sanitize_batch44_trace)
            + "\nmain()",
            {},
        )
    output = capsys.readouterr().out
    assert "private" not in output and "secret" not in output
    return json.loads(output), opener


def test_es_batch44_fixed_server_filters_and_sanitized_split_trace(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    report, opener = execute_es_diagnostic(es_trace_response(), capsys)
    assert (
        report["status"] == "trace_found" and report["exception_class"] == "ValueError"
    )
    assert report["frames"] == [
        {"filename": "onyx/regulatory/tasks.py", "function": "run", "line": 235}
    ]
    request = opener.open.call_args.args[0]
    assert (
        request.full_url.startswith("https://elastic.dev.singlewindow.io:9200/_search?")
        and request.method == "POST"
    )
    query = json.loads(request.data)
    assert query["size"] == 2000 and query["track_total_hits"] is True
    assert len(query["query"]["bool"]["filter"]) == 3
    assert cutover.NAMESPACE in json.dumps(query["query"]["bool"]["filter"][0])
    assert "dev-customs-regulations-background-.*" in json.dumps(
        query["query"]["bool"]["filter"][1]
    )
    assert query["_source"] == [
        "@timestamp",
        "kubernetes.namespace_name",
        "kubernetes.pod_name",
        "log",
        "message",
    ]


@pytest.mark.parametrize(
    "mutation,status",
    [
        ("namespace", "source_scope_mismatch"),
        ("time", "source_scope_mismatch"),
        ("pod", "source_scope_mismatch"),
        ("partial", "search_incomplete"),
        ("shards", "search_incomplete"),
        ("timeout", "search_incomplete"),
        ("different_pods", "batch_trace_unavailable"),
    ],
)
def test_es_batch44_refuses_scope_or_incomplete_proof(
    mutation: str, status: str, capsys: pytest.CaptureFixture[str]
) -> None:
    data = es_trace_response()
    source = data["hits"]["hits"][0]["_source"]
    if mutation == "namespace":
        source["kubernetes"]["namespace_name"] = "customs-regulations-test"
    if mutation == "time":
        source["@timestamp"] = "2026-09-11T10:50:00Z"
    if mutation == "pod":
        source["kubernetes"]["pod_name"] = "another-background"
    if mutation == "partial":
        data["hits"]["total"]["value"] += 1
    if mutation == "shards":
        data["_shards"]["failed"] = 1
    if mutation == "timeout":
        data["timed_out"] = True
    if mutation == "different_pods":
        data["hits"]["hits"][1]["_source"]["kubernetes"]["pod_name"] = (
            "dev-customs-regulations-background-b"
        )
    report, _ = execute_es_diagnostic(data, capsys)
    assert report["status"] == status


@pytest.mark.parametrize(
    "host,authorized", [("elastic.dev.singlewindow.io", True), ("other.example", False)]
)
def test_es_batch44_credentials_require_positive_endpoint_match(
    host: str, authorized: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    _, opener = execute_es_diagnostic(
        es_trace_response(),
        capsys,
        env={
            "ELASTICSEARCH_HOST": host,
            "ELASTICSEARCH_REST_API_PORT": "9200",
            "ELASTICSEARCH_USE_SSL": "true",
            "ELASTICSEARCH_ADMIN_USERNAME": "private",
            "ELASTICSEARCH_ADMIN_PASSWORD": "secret",
        },
    )
    assert bool(opener.open.call_args.args[0].get_header("Authorization")) == authorized


def test_es_batch44_cannot_redirect_or_bypass_tls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import inspect
    import ssl
    import urllib.request

    namespace: dict[str, Any] = {}
    exec(
        cutover.BATCH44_ES_PROGRAM
        + "\n"
        + inspect.getsource(cutover.sanitize_batch44_trace),
        namespace,
    )
    assert (
        namespace["NoRedirect"]().redirect_request(
            None, None, 302, "", {}, "https://other.example"
        )
        is None
    )
    with patch(
        "ssl.create_default_context", wraps=ssl.create_default_context
    ) as context:
        execute_es_diagnostic(es_trace_response(), capsys)
    context.assert_called_once_with(cafile=None)
    assert issubclass(namespace["NoRedirect"], urllib.request.HTTPRedirectHandler)


@pytest.mark.parametrize("code", [301, 401, 403])
def test_es_batch44_http_failures_emit_only_fixed_status(
    code: int, capsys: pytest.CaptureFixture[str]
) -> None:
    import inspect
    import json
    from urllib.error import HTTPError

    opener = Mock()
    opener.open.side_effect = HTTPError(
        "https://elastic.dev.singlewindow.io:9200", code, "secret", Message(), None
    )
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("urllib.request.build_opener", return_value=opener),
    ):
        exec(
            cutover.BATCH44_ES_PROGRAM
            + "\n"
            + inspect.getsource(cutover.sanitize_batch44_trace)
            + "\nmain()",
            {},
        )
    output = capsys.readouterr().out
    assert "secret" not in output and json.loads(output)["status"] == f"http_{code}"


def test_es_batch44_other_runtime_never_executes() -> None:
    driver = Mock(spec=cutover.Driver)
    driver.sha = "a" * 40
    cutover.diagnose_batch44_elasticsearch(driver, "pod", "container")
    driver.command.assert_not_called()


def test_worker_failure_receipt_survives_runner_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    detail = (
        '{"stage":"source_review","exceptions":[{"type":"RuntimeError","frames":[]}]}'
    )
    report = {
        "phase": "canary",
        "status": "failed",
        "release_sha_metadata": "a" * 40,
        "canary": {
            "evidence": {"worker_failure": detail},
            "retained_objects": [
                {
                    "kind": "worker_failure_receipt",
                    "id": "regulatory_amendment_failure:44:3",
                }
            ],
        },
    }
    with pytest.raises(cutover.CutoverRefusal, match="fixed_acceptance_probe_failed"):
        cutover.emit_acceptance_report(json.dumps(report), "canary", "a" * 40)
    assert json.loads(capsys.readouterr().out)["canary"] == report["canary"]


def test_comparison_diagnostic_sanitizes_validation_shape() -> None:
    import json

    from pydantic import ValidationError

    from onyx.regulatory.amendments.annexes.models import AnnexComparisonProposal

    payload = {
        "changes": [
            {
                "operation": "replace",
                "old_positions": [5],
                "new_positions": [],
                "explanation": "SECRET",
            }
        ],
        "old_positions": [5],
        "new_positions": [9],
        "old_pages": [2],
        "new_pages": [2],
        "SECRET_KEY": "SECRET_VALUE",
    }
    with pytest.raises(ValidationError) as captured:
        AnnexComparisonProposal.model_validate(payload)
    result = cutover.comparison_diagnostic_summary(
        json.dumps(payload), captured.value, "stop"
    )
    encoded = json.dumps(result)
    assert "SECRET" not in encoded
    assert result["finish_reason"] == "stop"
    assert result["change_shapes"][0]["old_positions"] == 1
    assert result["change_shapes"][0]["new_positions"] == 0
    assert any(
        error["validator"] == "invalid_operation_shape" for error in result["errors"]
    )
    assert result["unknown_fields"] == 1


def test_comparison_diagnostic_only_exact_runtime_and_readonly_wrapper() -> None:
    driver = Mock(spec=cutover.Driver)
    driver.sha = "a" * 40
    cutover.diagnose_batch47_comparison(driver, "pod", "container")
    driver.command.assert_not_called()
    driver.sha = cutover.BATCH47_RUNTIME
    driver.command.return_value = (
        '{"stage":"batch47_comparison","status":"scope_refused"}'
    )
    cutover.diagnose_batch47_comparison(driver, "pod", "container")
    command = driver.command.call_args.args[0]
    assert "PGOPTIONS='-c default_transaction_read_only=on'" in command[-3]
    assert "signal.alarm(180)" in command[-1]
    assert "provider_max_attempts" not in command[-1]
    assert driver.command.call_args.kwargs["timeout"] == 200


@pytest.mark.parametrize(
    "failure",
    [None, "review_hash", "page_hash", "scope", "provider", "snapshot", "missing_new"],
)
def test_fixed_comparison_replay_uses_readonly_dal_and_production_comparator(
    failure: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib
    import json
    from contextlib import nullcontext
    from io import BytesIO
    from types import SimpleNamespace
    from uuid import UUID, uuid4

    from onyx.regulatory.amendments.annexes import comparison, context_dependencies
    from onyx.regulatory.amendments.annexes.evidence import select_annex_evidence_view
    from onyx.regulatory.amendments.annexes.models import (
        AnnexChangeDraft,
        AnnexLocator,
        AnnexModelSnapshot,
        AnnexOriginalEvidence,
        AnnexReviewEvidence,
    )
    from tests.unit.onyx.regulatory.annexes.test_comparison import page
    from tests.unit.onyx.regulatory.annexes.test_evidence_views import mixed_pdf

    monkeypatch.setenv("POSTGRES_DB", "customs-regulations-dev")
    monkeypatch.setenv("REGULATORY_ANNEX_ENVIRONMENT", "dev")
    scope = {
        "batch_id": 47,
        "document_set_id": 21,
        "user_file_id": "e0acea4f-30f6-4b05-9bbf-5909b82ef0ab",
        "created_by": "7e0d56bc-6f9c-4cec-b29b-5a8c9ec2b844",
        "environment": "dev",
    }
    hashes = [
        "2a3035058c5a11366604b7e89052a44a7085eb54c705fd3cf09edc1c69f14ffe",
        "c5a20c8fd76d3dbd4983ceeaacc44eef217716427adec8716475ac5a753a0557",
    ]
    views, pages, evidence, files = [], [], [], {}
    for side, digest in zip(("old", "new"), hashes):
        original = AnnexOriginalEvidence(
            file_id=side, sha256=digest, mime_type="application/pdf", available=True
        )
        extracted = mixed_pdf().model_copy(update={"source_sha256": digest})
        view = select_annex_evidence_view(
            extraction=extracted,
            original=original,
            annex_label="EK-1",
            canonical_labels=["EK-1", "EK-2"],
            canonical_chunk_ids=["chunk"],
        )
        assert view.evidence_view is not None
        views.append(view)
        rendered = []
        for mapping in view.evidence_view.pages:
            image = page(number=mapping.view_page)
            rendered.append(image)
            file_id = side + str(mapping.view_page)
            files[file_id] = image.png
            evidence.append(
                AnnexReviewEvidence(
                    id=uuid4(),
                    side=side,
                    kind="comparison_page",
                    file_id=file_id,
                    sha256=hashlib.sha256(image.png).hexdigest(),
                    mime_type="image/png",
                    byte_count=len(image.png),
                    parent_file_id=side,
                    parent_sha256=digest,
                    locator=AnnexLocator(
                        page=mapping.original_page,
                        original_width=200,
                        original_height=200,
                    ),
                )
            )
        pages.append(rendered)
    retained = comparison.compare_annexes(
        old=views[0], new=views[1], old_pages=pages[0], new_pages=pages[1]
    ).model_copy(
        update={
            "model_snapshot": AnnexModelSnapshot(
                model_provider="vertex_ai", model_name="gemini-3.8-flash"
            )
        }
    )
    package_id = UUID("359c0315-e8ff-4233-8c53-fb5d17cc16ef")
    manifest = "1679e807427c324ed9c98bc643cd4f648e7055be28defa4e5f91b93042dee16f"
    draft = AnnexChangeDraft(
        instruction_indices=[0],
        instruction_texts=["Fictional instruction"],
        annex_label="EK-1",
        user_file_id=UUID(scope["user_file_id"]),
        source_package_id=package_id,
        source_manifest_sha256=manifest,
        old_extraction=views[0],
        new_extraction=views[1],
        comparison=retained,
        evidence=evidence,
    )
    payload = draft.model_dump(mode="json")
    review_hash = "92cfd62f7e29917003afd9a1b76be662d758903f57a5c3f94f02c8d50585cd08"
    row = SimpleNamespace(
        batch_id=47,
        environment="dev",
        user_file_id=draft.user_file_id,
        review_revision=1,
        publication_generation=0,
        status="blocked",
        review_sha256="bad" if failure == "review_hash" else review_hash,
        review_payload=payload,
    )
    batch = SimpleNamespace(
        document_set_id=22 if failure == "scope" else 21,
        source_package_id=package_id,
        created_by=UUID(scope["created_by"]),
    )
    package = SimpleNamespace(
        created_by=batch.created_by,
        idempotency_key="annex-canary-f5c14cd7-fae3-4b13-a67b-15e7eda68c92",
        manifest_sha256=manifest,
    )
    store = Mock()
    store.read_file_record.return_value = SimpleNamespace(
        file_metadata={"annex_review_scope": scope}
    )
    store.read_file.side_effect = lambda key: BytesIO(
        b"changed" if failure == "page_hash" else files[key]
    )
    llm = Mock()
    llm.config.model_provider = "openrouter" if failure == "provider" else "vertex_ai"
    llm.config.model_name = "other" if failure == "snapshot" else "gemini-3.8-flash"
    response = {
        "changes": [],
        "old_positions": comparison._atomic_positions(views[0]),
        "new_positions": comparison._atomic_positions(views[1]),
        "old_pages": [1, 2],
        "new_pages": [1, 2],
    }
    if failure == "missing_new":
        response["changes"] = [
            {
                "operation": "replace",
                "old_positions": [3],
                "new_positions": [],
                "explanation": "SECRET_VALUE",
            }
        ]
    llm.invoke.return_value = SimpleNamespace(
        choice=SimpleNamespace(finish_reason="stop")
    )
    real_hash = context_dependencies.context_hash
    with (
        patch("onyx.utils.variable_functionality.set_is_ee_based_on_env_variable"),
        patch(
            "onyx.db.regulatory_annex_dev_cutover.configured_indices",
            return_value=["owned_dev_index"],
        ),
        patch(
            "onyx.db.engine.sql_engine.SqlEngine.scoped_engine",
            return_value=nullcontext(),
        ) as engine,
        patch(
            "onyx.db.engine.sql_engine.get_session_with_current_tenant",
            side_effect=lambda: nullcontext(Mock()),
        ),
        patch("onyx.db.regulatory_amendments.get_batch", return_value=batch),
        patch("onyx.db.amendment_sources.get_source_package", return_value=package),
        patch(
            "onyx.db.regulatory_annex_changes.list_annex_changes", return_value=[row]
        ),
        patch.object(
            context_dependencies,
            "context_hash",
            side_effect=lambda value: (
                review_hash if value == payload else real_hash(value)
            ),
        ),
        patch("onyx.file_store.file_store.get_default_file_store", return_value=store),
        patch(
            "onyx.llm.factory.get_default_llm_with_vision", return_value=llm
        ) as factory,
        patch(
            "onyx.regulatory.structured_llm.llm_response_to_string",
            return_value=json.dumps(response),
        ),
        patch(
            "onyx.regulatory.structured_llm.llm_generation_span",
            return_value=nullcontext(Mock()),
        ),
        patch("onyx.regulatory.structured_llm.record_llm_response"),
    ):
        result = cutover.replay_batch47_comparison()
    assert "SECRET_VALUE" not in json.dumps(result)
    engine.assert_called_once_with(
        pool_size=2,
        max_overflow=0,
        connect_args={
            "options": "-c default_transaction_read_only=on",
            "connect_timeout": 10,
        },
    )
    if failure in {"review_hash", "scope", "page_hash"}:
        assert result["status"] == "scope_refused"
        llm.invoke.assert_not_called()
        factory.assert_not_called()
    elif failure in {"provider", "snapshot"}:
        assert result["status"] == "provider_refused"
        llm.invoke.assert_not_called()
    else:
        factory.assert_called_once_with()
        assert llm.invoke.call_count == (2 if failure == "missing_new" else 1)
        assert result["status"] == ("blocked" if failure == "missing_new" else "ready")
        assert result["page_sha256"] == [item.sha256 for item in evidence]
        assert result["model_name"] == "gemini-3.8-flash"
        assert len(result["attempts"]) == llm.invoke.call_count
        assert all(
            0 < call.kwargs["timeout_override"] <= 150
            for call in llm.invoke.call_args_list
        )
    assert not store.save_file.called


def test_comparison_diagnostic_embedded_failure_preserves_safe_details(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import inspect
    import json
    import logging

    driver = Mock(spec=cutover.Driver)
    driver.sha = cutover.BATCH47_RUNTIME
    driver.command.return_value = (
        '{"stage":"batch47_comparison","status":"scope_refused"}'
    )
    cutover.diagnose_batch47_comparison(driver, "pod", "container")
    capsys.readouterr()
    program = driver.command.call_args.args[0][-1]
    # Execute the exact wrapper failure path without DB, provider, or alarm activity.
    program = program.replace(
        inspect.getsource(cutover.replay_batch47_comparison),
        'def replay_batch47_comparison(report):\n    raise ValueError("SECRET_VALUE")\n',
    )
    monkeypatch.setattr("signal.signal", Mock())
    monkeypatch.setattr("signal.alarm", Mock())
    monkeypatch.setattr(logging, "disable", Mock())
    exec(program, {"__name__": "__main__"})
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "failed"
    assert "SECRET_VALUE" not in report["failure_detail"]
    assert json.loads(report["failure_detail"])["exceptions"][0]["type"] == "ValueError"
    assert report["attempts"] == []


@pytest.mark.parametrize(
    "extra",
    [
        {"secret": "value"},
        {"model_name": "SECRET"},
        {"attempts": [{"fields": ["SECRET"]}]},
    ],
)
def test_comparison_diagnostic_transport_refuses_unbounded_fields(
    extra: dict[str, Any],
) -> None:
    import json

    driver = Mock(spec=cutover.Driver)
    driver.sha = cutover.BATCH47_RUNTIME
    driver.command.return_value = json.dumps(
        {"stage": "batch47_comparison", "status": "blocked", **extra}
    )
    with pytest.raises(RuntimeError, match="fixed_diagnostic_report_required"):
        cutover.diagnose_batch47_comparison(driver, "pod", "container")


def test_source_package_failure_survives_canary_transport(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    detail = (
        '{"stage":"source_package","exceptions":[{"type":"ValueError","frames":[]}]}'
    )
    report = {
        "phase": "canary",
        "status": "failed",
        "release_sha_metadata": "a" * 40,
        "evidence": {"source_package_status": "failed", "worker_failure": detail},
    }
    with pytest.raises(cutover.CutoverRefusal, match="fixed_acceptance_probe_failed"):
        cutover.emit_acceptance_report(json.dumps(report), "canary", "a" * 40)
    assert json.loads(capsys.readouterr().out)["evidence"] == report["evidence"]
    report["evidence"]["source_package_status"] = "SECRET_INPUT"
    with pytest.raises(
        cutover.CutoverRefusal, match="acceptance_source_package_status_refused"
    ):
        cutover.emit_acceptance_report(json.dumps(report), "canary", "a" * 40)


@pytest.mark.parametrize("migration", [False, True])
def test_protected_redis_migration_quiesces_without_rebarrier(migration: bool) -> None:
    driver = Mock(spec=cutover.Driver)
    driver.continue_protected_release.return_value = True
    driver.redis_isolation.return_value = migration
    cutover.prepare(driver)
    expected = ["validate_target", "continue_protected_release", "redis_isolation"]
    if migration:
        expected += [
            "record_started",
            "stop_api",
            "drain_background",
            "stop_background",
            "assert_no_writers",
        ]
    assert [call[0] for call in driver.method_calls] == expected + ["record_installing"]
    driver.redis_isolation.assert_called_once_with("reserve")
    driver.block_indices.assert_not_called()


def test_redis_reservation_refusal_prevents_any_stop() -> None:
    driver = Mock(spec=cutover.Driver)
    driver.continue_protected_release.return_value = True
    driver.redis_isolation.side_effect = cutover.CutoverRefusal("occupied")
    with pytest.raises(cutover.CutoverRefusal):
        cutover.prepare(driver)
    driver.stop_api.assert_not_called()
    driver.drain_background.assert_not_called()
    driver.record_installing.assert_not_called()


def test_redis_helper_injection_is_fixed_and_parses_exact_receipt() -> None:
    driver = cutover.Driver("a" * 40)
    with (
        patch.object(driver, "pods", return_value=[runtime_pod()]),
        patch.object(Path, "read_text", return_value="# fixed helper") as source,
        patch.object(
            driver,
            "command",
            return_value='{"redis_isolation":"reserved","migration_required":true}',
        ) as command,
    ):
        assert driver.redis_isolation("reserve") is True
    source.assert_called_once()
    assert command.call_args.args[1] == "# fixed helper"
    args = command.call_args.args[0]
    assert args[-3:] == [
        '. /vault/secrets/config; exec python - "$1"',
        "redis-isolation",
        "reserve",
    ]
    assert "-i" in args
    assert command.call_args.kwargs["timeout"] == 60


@pytest.mark.parametrize(
    "receipt",
    [
        '{"redis_isolation":"verified","migration_required":true}',
        '{"redis_isolation":"verified","migration_required":false,"extra":"secret"}',
        '{"redis_isolation":"reserved","migration_required":false}',
    ],
)
def test_redis_verification_refuses_wrong_receipt(receipt: str) -> None:
    driver = cutover.Driver("a" * 40)
    with (
        patch.object(driver, "pods", return_value=[runtime_pod()]),
        patch.object(Path, "read_text", return_value="helper"),
        patch.object(driver, "command", return_value=receipt),
        pytest.raises(cutover.CutoverRefusal),
    ):
        driver.redis_isolation("verify", app="api")


def test_runtime_redis_checks_after_settled_apps_only_once() -> None:
    driver = cutover.Driver("a" * 40)
    events = Mock()
    with (
        patch.object(driver, "verify_app", events.verify_app),
        patch.object(driver, "redis_isolation", events.redis_isolation),
    ):
        driver.verify_runtime()
    assert [(call[0], call.args, call.kwargs) for call in events.method_calls] == [
        ("verify_app", ("api",), {"readiness": True}),
        ("verify_app", ("background",), {"readiness": True}),
        ("redis_isolation", ("verify",), {"app": "api"}),
        ("redis_isolation", ("verify",), {"app": "background"}),
        ("redis_isolation", ("worker",), {}),
    ]


def test_redis_deployment_values_override_both_apps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    monkeypatch.setenv(
        "REGULATORY_LABELING_VERTEX_GCS_URI", "gs://labeling-test-bucket/staging"
    )
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "devops/dev/customs-regulations"
    directory.mkdir(parents=True)
    for app in cutover.APPS:
        (directory / f"customs-regulations-{app}-values.yaml").write_text(
            yaml.safe_dump(
                {
                    "app": {
                        "environment": {
                            "parameters": [
                                {
                                    "name": "REDIS_DEPLOYMENT_DATABASES",
                                    "value": "0,15,14",
                                }
                            ]
                        }
                    }
                }
            )
        )
    cutover.render_values()
    for app in cutover.APPS:
        data = yaml.safe_load(
            (directory / f"customs-regulations-{app}-values.yaml").read_text()
        )
        assert [
            entry
            for entry in data["app"]["environment"]["parameters"]
            if entry["name"] == "REDIS_DEPLOYMENT_DATABASES"
        ] == [{"name": "REDIS_DEPLOYMENT_DATABASES", "value": "4,5,6"}]


def test_standard_dev_deploy_renders_configuration_and_probes_actual_worker() -> None:
    import yaml

    root = Path(__file__).resolve().parents[4]
    workflow = yaml.safe_load(
        (
            root / ".github/workflows/customs-regulations-backend-lite-codebuild.yaml"
        ).read_text()
    )
    steps = workflow["jobs"]["build-and-deploy"]["steps"]
    render = next(
        step
        for step in steps
        if step["name"] == "Render required DEV runtime configuration"
    )
    deploy = next(step for step in steps if step.get("id") == "helm_deploy")
    verify = next(
        step
        for step in steps
        if step["name"] == "Verify DEV annex consumer and labeling configuration"
    )
    assert render["if"] == "env.env_x == 'dev'"
    assert "render_values(enabled=True)" in render["run"]
    assert steps.index(render) < steps.index(deploy) < steps.index(verify)
    assert "regulatory_annex_readiness" in verify["run"]
    assert "_configured_vertex_staging_uri" in verify["run"]
    assert ". /vault/secrets/config" in verify["run"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "gcs_uri", [None, "", "gs://labeling-test-bucket/native-batch"]
)
def test_labeling_vertex_storage_values_match_both_apps_and_replace_stale_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    gcs_uri: str | None,
) -> None:
    import yaml

    name = "REGULATORY_LABELING_VERTEX_GCS_URI"
    monkeypatch.chdir(tmp_path)
    if gcs_uri is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, gcs_uri)
    directory = tmp_path / "devops/dev/customs-regulations"
    directory.mkdir(parents=True)
    for app in cutover.APPS:
        parameters = (
            [{"name": name, "value": "gs://stale-test-bucket/old-prefix"}]
            if app == "api"
            else []
        )
        (directory / f"customs-regulations-{app}-values.yaml").write_text(
            yaml.safe_dump({"app": {"environment": {"parameters": parameters}}})
        )

    if not gcs_uri:
        originals = {path: path.read_text() for path in directory.iterdir()}
        with pytest.raises(cutover.CutoverRefusal, match="GCS_location_required"):
            cutover.render_values(enabled)
        assert all(path.read_text() == original for path, original in originals.items())
        return
    cutover.render_values(enabled)
    cutover.render_values(enabled)

    for app in cutover.APPS:
        data = yaml.safe_load(
            (directory / f"customs-regulations-{app}-values.yaml").read_text()
        )
        parameters = data["app"]["environment"]["parameters"]
        assert [entry for entry in parameters if entry["name"] == name] == [
            {"name": name, "value": gcs_uri or ""}
        ]
        assert (
            next(
                entry["value"]
                for entry in parameters
                if entry["name"] == "REGULATORY_ANNEX_UPDATES_ENABLED"
            )
            == str(enabled).lower()
        )


def test_labeling_vertex_storage_refuses_duplicate_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    name = "REGULATORY_LABELING_VERTEX_GCS_URI"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(name, "gs://labeling-test-bucket/native-batch")
    directory = tmp_path / "devops/dev/customs-regulations"
    directory.mkdir(parents=True)
    paths = []
    for app in cutover.APPS:
        path = directory / f"customs-regulations-{app}-values.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "app": {
                        "environment": {
                            "parameters": [
                                {"name": name, "value": "gs://first-test-bucket/path"},
                                {"name": name, "value": "gs://second-test-bucket/path"},
                            ]
                        }
                    }
                }
            )
        )
        paths.append(path)
    original = [path.read_text() for path in paths]

    with pytest.raises(cutover.CutoverRefusal, match="duplicate_environment_parameter"):
        cutover.render_values()

    assert [path.read_text() for path in paths] == original


def test_redis_failure_keeps_only_fixed_safe_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = cutover.Driver("a" * 40)
    args = [
        "kubectl",
        "--namespace",
        cutover.NAMESPACE,
        "exec",
        "pod",
        "--",
        "sh",
        "-c",
        '. /vault/secrets/config; exec python - "$1"',
        "redis-isolation",
        "verify",
    ]
    with (
        patch.object(
            cutover.subprocess,
            "run",
            return_value=Mock(
                returncode=1,
                stdout="NOT_READY redis_isolation OwnerMismatch\nsecret body\nNOT_READY redis_isolation secret credential\n",
                stderr="secret credential",
            ),
        ),
        pytest.raises(cutover.CutoverRefusal),
    ):
        driver.command(args, "fixed-helper", timeout=60)
    output = capsys.readouterr().out
    assert "NOT_READY redis_isolation OwnerMismatch" in output
    assert "secret" not in output


def test_activation_does_not_repeat_already_verified_background_worker() -> None:
    driver = Mock(spec=cutover.Driver)
    driver.sha = "a" * 40
    driver.get.return_value = {"data": {"sha": driver.sha, "phase": "released"}}
    with (
        patch.object(cutover, "require_release_runs"),
        patch.object(cutover, "verify_frontend"),
        patch.object(cutover, "acceptance"),
        patch.object(cutover, "deploy_same_image"),
    ):
        cutover.verify_or_activate(driver, True)
    assert driver.verify_runtime.call_args_list[-1].kwargs == {"redis_worker": False}


def test_failed_migration_drain_keeps_new_transition_for_failure_handler() -> None:
    import json

    driver = cutover.Driver("a" * 40)
    persisted: dict[str, Any] = {"data": {"sha": "b" * 40, "phase": "released"}}

    def kubectl(*args: str, stdin: str | None = None) -> str:
        assert args == ("apply", "-f", "-") and stdin is not None
        persisted.clear()
        persisted.update(json.loads(stdin))
        return ""

    with (
        patch.object(driver, "validate_target"),
        patch.object(driver, "continue_protected_release", return_value=True),
        patch.object(driver, "redis_isolation", return_value=True),
        patch.object(driver, "kubectl", side_effect=kubectl),
        patch.object(driver, "get", side_effect=lambda *_args: persisted),
        patch.object(driver, "stop_api") as stop,
        patch.object(
            driver, "drain_background", side_effect=RuntimeError("drain failed")
        ) as drain,
        patch.object(driver, "stop_background") as background,
        patch.object(driver, "record_installing") as install,
    ):
        with pytest.raises(RuntimeError, match="drain failed"):
            cutover.prepare(driver)
        assert persisted["data"]["sha"] == driver.sha
        assert persisted["data"]["phase"] == "quiescing"
        install.assert_not_called()
        background.assert_not_called()
        drain.side_effect = None
        driver.failure()
        assert stop.call_count == 2
        background.assert_called_once()


@pytest.mark.parametrize("missing_node", [False, True])
def test_protected_release_captures_current_pod_nodes(missing_node: bool) -> None:
    import json

    driver = cutover.Driver("a" * 40)
    state = {
        "metadata": {"name": cutover.STATE},
        "data": {
            "sha": "b" * 40,
            "phase": "released",
            "indices": '{"index":"uuid"}',
            "scope": '{"ownership":"retained"}',
            "container": "main",
            "writer_nodes": json.dumps(["terminated-original-node"]),
        },
    }
    pods = [runtime_pod(sha="b" * 40), runtime_pod(sha="b" * 40)]
    pods[0]["spec"]["nodeName"] = "current-api-node"
    if not missing_node:
        pods[1]["spec"]["nodeName"] = "current-background-node"
    with (
        patch.object(driver, "get", return_value={"items": [state]}),
        patch.object(driver, "pods", side_effect=[[pod] for pod in pods]),
    ):
        if missing_node:
            with pytest.raises(
                cutover.CutoverRefusal, match="current_writer_node_required"
            ):
                driver.continue_protected_release()
        else:
            assert driver.continue_protected_release() is True
            assert driver.writer_nodes == {
                "current-api-node",
                "current-background-node",
            }
            assert driver.indices == {"index": "uuid"}
            assert driver.scope == {"ownership": "retained"}
