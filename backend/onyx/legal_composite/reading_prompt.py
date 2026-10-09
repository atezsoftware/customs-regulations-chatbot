"""Use short delivered selectors only for the opt-in source-reading response."""

from onyx.legal_composite.prompts import COMMON, RESEARCH_PROMPT

_LEGACY_SELECTOR = COMMON[
    COMMON.index("Complete original_evidence contains witness_spans:") :
]
_NUMBERED_SELECTOR = """Complete original_evidence contains witness_spans with one-based span_number,
start_char and end_char into the complete original text. For each source-reading
requirement support, return only citation and span_number from that citation's provided
witness catalogue. Do not return span_id or quotation. Read the original before selecting
spans; the host binds the chosen number to that exact delivered original passage without
another model call. Select multiple adjacent spans when decisive conditions cross a
boundary. Never select a number from another citation, an omitted original or a merely
similar passage. This selector binds evidence; it does not prove applicability.
"""
if RESEARCH_PROMPT.count(_LEGACY_SELECTOR) != 1:
    raise ValueError(
        "Source-reading selector instructions must have one exact location"
    )

NUMBERED_READING_PROMPT = RESEARCH_PROMPT.replace(
    _LEGACY_SELECTOR, _NUMBERED_SELECTOR, 1
)
_LEGACY_READING_SELECTOR = (
    "Use provided span_id references instead of retyping original quotations."
)
if NUMBERED_READING_PROMPT.count(_LEGACY_READING_SELECTOR) != 1:
    raise ValueError("Requirement selector instructions must have one exact location")
NUMBERED_READING_PROMPT = NUMBERED_READING_PROMPT.replace(
    _LEGACY_READING_SELECTOR,
    "Use provided citation/span_number pairs instead of retyping original quotations.",
    1,
)
