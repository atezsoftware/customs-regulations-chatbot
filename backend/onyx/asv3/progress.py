from __future__ import annotations

import hashlib
import json
import re
import threading
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

from pydantic import BaseModel, JsonValue

if TYPE_CHECKING:
    from onyx.asv3.evidence import EvidenceLedger
    from onyx.asv3.models import EvidenceItem, RunContext


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


def _source_heading(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    heading = " ".join(value.strip(" #\t\r\n").split())
    without_numbers = re.sub(r"\b\d{1,4}/\d{1,4}\b", "", heading)
    if (
        not heading
        or len(heading) > 200
        or any(part in without_numbers for part in ("/", "\\", "_"))
        or re.search(r"\.(?:md|pdf|docx?|txt|html?)\b|https?://", heading, re.I)
    ):
        return None
    return heading


def _canonical_corpus_original(item: "EvidenceItem") -> bool:
    from onyx.asv3.models import model_evidence_metadata

    metadata = model_evidence_metadata(item.metadata)
    return bool(
        item.chunk_id
        and item.search_doc is not None
        and item.search_doc.document_id == item.source_id
        and item.search_doc.metadata.get("regulatory_chunk_id") == item.chunk_id
        and not metadata.get("derived")
        and not metadata.get("external")
    )


def official_corpus_source_name(item: "EvidenceItem") -> str | None:
    """Return a verified presentation name without changing canonical source identity."""
    from onyx.asv3.models import model_evidence_metadata

    if not _canonical_corpus_original(item):
        return None
    metadata = model_evidence_metadata(item.metadata)
    raw_headings = metadata.get("heading_path")
    headings = raw_headings if isinstance(raw_headings, list) else []
    if str(metadata.get("document_type", "")).lower() == "genelge":
        for heading in headings:
            if not isinstance(heading, str):
                continue
            match = re.fullmatch(r"\s*\(?\s*(\d{4}/\d{1,4})\s*\)?\s*", heading)
            if match:
                return f"{match.group(1)} sayılı Genelge"
    source_type = re.compile(
        r"\b(?:kanun(?:u|un|unun)?|yönetmeli(?:k|ği)|tebli(?:ğ|ği)|genelge(?:si)?|"
        r"karar(?:ı|name)?|tüzü(?:k|ğü)|sirküler(?:i)?|özelge|law|act|code|"
        r"regulation|directive|decision|circular)\b",
        re.I,
    )
    for candidate in [headings[0] if headings else None, metadata.get("title")]:
        name = _source_heading(candidate)
        if name and source_type.search(name):
            return name
    return None


def _readable_corpus_source_name(item: "EvidenceItem") -> str | None:
    from onyx.asv3.models import model_evidence_metadata

    metadata = model_evidence_metadata(item.metadata)
    for value in [
        metadata.get("title"),
        item.search_doc.semantic_identifier if item.search_doc else None,
    ]:
        if not isinstance(value, str) or re.search(r"https?://", value, re.I):
            continue
        title = value.split(" — ", 1)[0].strip()
        without_numbers = re.sub(r"\b\d{1,4}/\d{1,4}\b", "", title)
        if "/" in without_numbers or "\\" in without_numbers:
            title = re.split(r"[/\\]", title)[-1]
        title = re.sub(
            r"\.(?:md|pdf|docx?|txt|html?|rtf|odt|xlsx?|csv|pptx?)$",
            "",
            title,
            flags=re.I,
        )
        if re.fullmatch(r"(?:[a-f0-9-]{32,}|rc_[a-f0-9]+|\d+)", title, re.I):
            continue
        title = re.sub(r"_+", " ", title)
        name = _source_heading(title)
        if name and re.search(r"[^\W\d_]", name):
            return name
    return None


def _source_display(
    item: "EvidenceItem", language: str, fallback_title: str
) -> tuple[str, str | None, str | None]:
    from onyx.asv3.models import model_evidence_metadata
    from onyx.regulatory.heading_path import parse_regulatory_article_heading

    metadata = model_evidence_metadata(item.metadata)
    raw_headings = metadata.get("heading_path")
    headings = raw_headings if isinstance(raw_headings, list) else []
    article, qualifier, article_heading = None, None, None
    for heading in reversed(headings):
        if not isinstance(heading, str):
            continue
        parsed = parse_regulatory_article_heading(heading)
        if parsed:
            article, qualifier = parsed.article_no, parsed.qualifier
            article_heading = _source_heading(heading)
            break
    number = metadata.get("article_no")
    if article is None and isinstance(number, str):
        if parsed := parse_regulatory_article_heading("Madde " + number):
            article, qualifier = parsed.article_no, parsed.qualifier
    source = (
        official_corpus_source_name(item)
        or _readable_corpus_source_name(item)
        or fallback_title
    )
    if article:
        if language in {"tr", "en"} and not qualifier:
            source += f" — {'Madde' if language == 'tr' else 'Article'} {article}"
        elif article_heading:
            source += " — " + article_heading
    return source[:240], article, qualifier


def report_source_deliveries(
    ledger: "EvidenceLedger",
    call_id: str | None,
    context: "RunContext",
    reporter: "ProgressReporter",
    localized_tools: list[str],
) -> None:
    from onyx.asv3.supplemental_tools import public_narration_valid
    from onyx.tracing.flows import LLMFlow

    if (
        not call_id
        or context.depth
        or context.is_cancelled()
        or ledger.delivery_flow(call_id) != LLMFlow.ASV3_COORDINATOR.value
    ):
        return
    language = reporter.language.split("-")[0].lower()
    fallback_title, message = {
        "tr": (
            "İlgili hükümler inceleniyor",
            "Bu kaynaktaki ilgili özgün hükümler inceleniyor.",
        ),
        "en": (
            "Reviewing relevant provisions",
            "Reviewing the relevant original provisions in this source.",
        ),
    }.get(language, tuple(localized_tools))
    seen = {
        event.task_id
        for event in reporter.snapshot()
        if event.task_id and event.task_id.startswith("action:source:")
    }
    for citation in sorted(ledger.completely_delivered(call_id)):
        if context.is_cancelled():
            return
        item = ledger.get(citation)
        if item is None or not _canonical_corpus_original(item):
            continue
        title, article, qualifier = _source_display(item, language, fallback_title)
        identity = json.dumps([item.source_id, article, qualifier], ensure_ascii=False)
        task_id = "action:source:" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        if task_id in seen:
            continue
        if not public_narration_valid(title, message, context):
            title = fallback_title
        if not public_narration_valid(title, message, context):
            continue
        seen.add(task_id)
        reporter.report(
            "tools", status="completed", task_id=task_id, title=title, message=message
        )


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
            "Araştırma tamamlanamadı; mevcut bulgular korunuyor.",
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
            "Research could not be completed; existing findings are preserved.",
        ),
    },
}


def localized_notifications(language: str) -> dict[str, list[str]]:
    """Supply terminal UI text without a separate language-model invocation."""
    base = _MESSAGES.get(language.split("-")[0].lower(), {})
    result = {
        phase: list(base.get(phase, ("ASv3", "…")))
        for phase in (
            "started",
            "tools",
            "worker",
            "final",
            "completed",
            "failed",
            "cancelled",
            "interrupted",
            "resume",
            "native_citation",
        )
    }
    if language.startswith("tr"):
        result.update(
            final=["Yanıt hazırlanıyor", "Kaynakları verilen olaya uyguluyorum."],
            completed=["Yanıt hazır", "Kaynaklı yanıt tamamlandı."],
            interrupted=[
                "Araştırma yarım kaldı",
                "Araştırma tamamlanamadı; mevcut kanıtlar korunuyor.",
            ],
            resume=[
                "Araştırmaya devam",
                "Açık kalan hususları incelemeye devam edebilirim.",
            ],
            native_citation=[
                "Özgün kaynaktan pasaj",
                "Özgün dosyadan çıkarılan kaynak pasajı.",
            ],
        )
    elif language.startswith("en"):
        result.update(
            final=[
                "Preparing the answer",
                "I am applying the original sources to your facts.",
            ],
            completed=["Answer ready", "The source-grounded answer is ready."],
            interrupted=[
                "Research interrupted",
                "Research is incomplete; the evidence is retained.",
            ],
            resume=[
                "Continue research",
                "I can continue investigating the unresolved issues.",
            ],
            native_citation=[
                "Original source excerpt",
                "An excerpt extracted from the original file.",
            ],
        )
    return result


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
        localized = False
        for event in events:
            if (
                event.run_id != self.run_id
                or (
                    event.language != self.language
                    and not (event.language == "und" and not localized)
                )
                or (event.language == "und" and localized)
                or event.sequence <= previous
            ):
                raise ValueError("Invalid progress event sequence")
            localized = localized or event.language != "und"
            previous = event.sequence
        if sequence != previous:
            raise ValueError("Progress sequence does not match events")
        with self._lock:
            self._events, self._sequence = events, sequence
