from pathlib import Path


def test_backend_deploy_verifies_the_benchmark_worker_is_running() -> None:
    repository_root = Path(__file__).parents[6]
    workflow = (
        repository_root
        / ".github"
        / "workflows"
        / "customs-regulations-backend-lite-codebuild.yaml"
    ).read_text(encoding="utf-8")

    assert (
        'deploy_app "customs-regulations-background" '
        '"customs-regulations/customs-regulations-background-values.yaml" "true" "4Gi"'
        in workflow
    )
    assert 'local memory_limit="$4"' in workflow
    assert 'memory_override+=(--set "app.mem_limits=$memory_limit")' in workflow
    assert 'kubectl rollout status deployment "$deployment" \\' in workflow
    assert '--namespace "${namespace}" --timeout=600s' in workflow
    assert "verify_benchmark_worker" in workflow
    assert (
        "supervisorctl -c /etc/supervisor/conf.d/supervisord.conf status "
        "celery_worker_regulatory_benchmark"
    ) in workflow
    assert 'grep -q "RUNNING"' in workflow
    assert "/tmp/onyx_k8s_regulatorybenchmark_readiness.txt" in workflow
    assert 'test -f "$readiness_file"' in workflow
