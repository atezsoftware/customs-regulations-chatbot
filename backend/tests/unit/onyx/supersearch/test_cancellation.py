from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock

from onyx.supersearch.cancellation import CancellationProbe


def test_parallel_chunk_checks_share_one_stop_poll() -> None:
    cancelled = Mock(return_value=False)
    probe = CancellationProbe(cancelled, clock=lambda: 10.0)
    ready = Barrier(4)

    def check_rows() -> list[bool]:
        ready.wait(timeout=5)
        return [probe() for _ in range(1_000)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: check_rows(), range(4)))
    assert all(result == [False] * 1_000 for result in results)
    cancelled.assert_called_once_with()


def test_stop_poll_refreshes_and_retains_cancellation() -> None:
    now = [10.0]
    cancelled = Mock(side_effect=[False, True])
    probe = CancellationProbe(cancelled, clock=lambda: now[0])
    assert not probe()
    now[0] += 0.05
    assert not probe()
    assert cancelled.call_count == 1
    now[0] += 0.06
    assert probe()
    now[0] += 1
    assert probe(force=True)
    assert cancelled.call_count == 2


def test_final_publication_forces_a_fresh_stop_read() -> None:
    cancelled = Mock(side_effect=[False, True])
    probe = CancellationProbe(cancelled, clock=lambda: 10.0)
    assert not probe()
    assert probe(force=True)
    assert cancelled.call_count == 2


def test_runs_do_not_share_cached_stop_state() -> None:
    assert CancellationProbe(lambda: True)()
    assert not CancellationProbe(lambda: False)()
