import pytest

from onyx.asv3.progress import ProgressReporter


def test_resume_retains_neutral_start_before_first_action_language() -> None:
    reporter = ProgressReporter("native-run", "und")
    initial = reporter.report("started")
    reporter.language = "tr"
    first_action = reporter.report(
        "tools", title="Kaynaklar inceleniyor", message="İlgili hükmü okuyorum."
    )
    resumed = ProgressReporter("native-run", "tr")
    resumed.restore(reporter.export())
    continued = resumed.report("resume")

    assert [event.language for event in resumed.snapshot()] == ["und", "tr", "tr"]
    assert [event.sequence for event in resumed.snapshot()] == [1, 2, 3]
    assert initial.event_id == resumed.snapshot()[0].event_id
    assert first_action.event_id == resumed.snapshot()[1].event_id
    assert continued.language == "tr"


@pytest.mark.parametrize("foreign_language", ["en", "und"])
def test_resume_rejects_language_drift_after_localized_action(
    foreign_language: str,
) -> None:
    reporter = ProgressReporter("native-run", "tr")
    reporter.report("started")
    reporter.language = foreign_language
    reporter.report("tools")
    payload = reporter.export()
    payload["language"] = "tr"

    with pytest.raises(ValueError, match="Invalid progress event"):
        ProgressReporter("native-run", "tr").restore(payload)


def test_resume_still_rejects_a_different_profile_identity() -> None:
    reporter = ProgressReporter("native-run", "und")
    reporter.report("started")
    reporter.language = "tr"
    reporter.report("tools")

    with pytest.raises(ValueError, match="identity mismatch"):
        ProgressReporter("native-run", "en").restore(reporter.export())
