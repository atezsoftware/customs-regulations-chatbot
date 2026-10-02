from __future__ import annotations

import hashlib
import threading
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

from pydantic import BaseModel, JsonValue

if TYPE_CHECKING:
    from onyx.asv3.models import RunContext


class ProgressEvent(BaseModel):
    run_id: str
    event_id: str
    sequence: int
    language: str
    phase: str
    status: str
    title: str
    message: str
    task_id: str | None = None
    parent_task_id: str | None = None
    active_workers: int = 0
    completed_workers: int = 0
    public_narration: bool = False


def public_action_id(call_id: str) -> str:
    """Provider call IDs may contain private, large opaque signatures."""
    return "action:" + hashlib.sha256(call_id.encode()).hexdigest()[:24]


def action_narration(
    arguments: dict[str, JsonValue], context: "RunContext"
) -> tuple[str, str] | None:
    from onyx.asv3.supplemental_tools import public_narration_valid

    update = arguments.get("_public_update")
    if (
        not isinstance(update, list)
        or len(update) != 2
        or not all(isinstance(item, str) for item in update)
    ):
        return None
    title, message = update
    assert isinstance(title, str) and isinstance(message, str)
    return (title, message) if public_narration_valid(title, message, context) else None


_MESSAGES: dict[str, dict[str, tuple[str, str]]] = {
    "tr": {
        "started": ("Araştırma başladı", "Soruyu ve kaynakları değerlendiriyorum."),
        "tools": (
            "Kaynaklar inceleniyor",
            "Gerekli bilgi için uygun araçları kullanıyorum.",
        ),
        "worker": ("Paralel araştırma", "Bağımsız bir bilgi ihtiyacı araştırılıyor."),
        "completed": ("Araştırma tamamlandı", "Bulgular yanıt için hazır."),
        "cancelled": (
            "Araştırma durduruldu",
            "Çalışma durduruldu; geç sonuçlar kabul edilmeyecek.",
        ),
        "failed": (
            "Araştırma tamamlanamadı",
            "Çalışmanın sınırına ulaşıldı; mevcut bulgular korunuyor.",
        ),
    },
    "en": {
        "started": ("Research started", "I am assessing the question and its sources."),
        "tools": (
            "Reviewing sources",
            "I am using suitable tools to obtain the evidence.",
        ),
        "worker": (
            "Parallel research",
            "An independent information need is being investigated.",
        ),
        "completed": ("Research completed", "The findings are ready for the answer."),
        "cancelled": (
            "Research stopped",
            "The work has stopped; late results will not be accepted.",
        ),
        "failed": (
            "Research incomplete",
            "The run reached its limit; existing findings are preserved.",
        ),
    },
}


class ProgressReporter:
    def __init__(
        self,
        run_id: str,
        language: str,
        emit: Callable[[ProgressEvent], None] | None = None,
        translate: Callable[[str, str], tuple[str, str]] | None = None,
    ) -> None:
        self.run_id = run_id
        self.language = language
        self._emit = emit or (lambda _event: None)
        self._translate = translate
        self._sequence = 0
        self._lock = threading.RLock()
        self._events: list[ProgressEvent] = []

    def report(
        self,
        phase: str,
        *,
        status: str = "running",
        task_id: str | None = None,
        parent_task_id: str | None = None,
        active_workers: int = 0,
        completed_workers: int = 0,
        title: str | None = None,
        message: str | None = None,
    ) -> ProgressEvent:
        with self._lock:
            language = self.language.split("-")[0].lower()
            if language in _MESSAGES:
                fallback_title, fallback_message = _MESSAGES[language].get(
                    phase, _MESSAGES[language]["tools"]
                )
            elif self._translate:
                fallback_title, fallback_message = self._translate(phase, self.language)
            else:
                # Language-neutral fallback; adapters should supply translation for other languages.
                fallback_title, fallback_message = "…", "…"
            self._sequence += 1
            event = ProgressEvent(
                run_id=self.run_id,
                event_id=str(uuid4()),
                sequence=self._sequence,
                language=self.language,
                phase=phase,
                status=status,
                title=(title or fallback_title)[:240],
                message=(message or fallback_message)[:1600],
                task_id=task_id,
                parent_task_id=parent_task_id,
                active_workers=active_workers,
                completed_workers=completed_workers,
                public_narration=title is not None and message is not None,
            )
            self._events.append(event)
            if len(self._events) > 200:
                latest = {item.task_id or "coordinator": item for item in self._events}
                retained = {
                    item.event_id: item
                    for item in [
                        self._events[0],
                        *latest.values(),
                        *self._events[-100:],
                    ]
                }
                self._events = sorted(retained.values(), key=lambda item: item.sequence)
        # Callbacks may acquire checkpoint locks and export this reporter again.
        self._emit(event)
        return event

    def snapshot(self) -> list[ProgressEvent]:
        with self._lock:
            return [event.model_copy(deep=True) for event in self._events]

    def export(self) -> dict[str, JsonValue]:
        with self._lock:
            return {
                "version": 1,
                "run_id": self.run_id,
                "language": self.language,
                "sequence": self._sequence,
                "events": [event.model_dump(mode="json") for event in self._events],
            }

    def restore(self, payload: dict[str, JsonValue]) -> None:
        if (
            payload.get("version") != 1
            or payload.get("run_id") != self.run_id
            or payload.get("language") != self.language
        ):
            raise ValueError("Progress checkpoint identity mismatch")
        raw_events = payload.get("events")
        sequence = payload.get("sequence")
        if not isinstance(raw_events, list) or not isinstance(sequence, int):
            raise ValueError("Invalid progress checkpoint")
        events = [ProgressEvent.model_validate(item) for item in raw_events]
        previous = 0
        for event in events:
            if (
                event.run_id != self.run_id
                or event.language != self.language
                or event.sequence <= previous
            ):
                raise ValueError("Invalid progress event sequence")
            previous = event.sequence
        if sequence != previous:
            raise ValueError("Progress sequence does not match events")
        with self._lock:
            self._events, self._sequence = events, sequence
