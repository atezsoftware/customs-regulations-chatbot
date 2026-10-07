"""Source-lead follow-through for the isolated normal-profile experiment."""

from onyx.prompts.asv3.coordinator_reference import (
    COORDINATOR_REFERENCE_PROMPT,
    RESEARCHER_REFERENCE_PROMPT,
)
from onyx.prompts.asv3.research import LEGAL_DEPARTMENT_RESEARCH

TUNED_PROMPT_VERSION = "asv3-tuned-2026-10-07.17"

_SOURCE_ACTIONS = """For a known instrument and article, read_provision uses the supplied source_id;
otherwise read_named_provision resolves its own title and reads that article in one action.
Do not search the whole corpus merely to read a known provision: the same article number
in another instrument is not its original. Ambiguous identities require clarification of
the source, not choosing the first candidate. Research genuinely unresolved effects,
exceptions and related authorities separately with their own focused searches. Retain
source_id for source-local searches and continuation reads. An explicit single-instrument
article target uses canonical reading even in search_corpus; set discover_related_sources
true when seeking other authorities rather than that provision's own text."""

_METADATA_DELTAS = """Outcome metadata is retained by the host. _outcomes and _coverage carry changes only;
omit unchanged records or use null where the schema requires the field. Keep retained
condition IDs, requirements and witnesses unchanged. Add genuinely new requirements with
new IDs, and update outcome resolutions independently; do not repeat the whole catalogue."""


def _focused_reference(prompt: str) -> str:
    before, separator, after = prompt.partition(
        "For a known source/article use resolve_source/read_provision; retain source_id in source-local\nsearches and fallbacks."
    )
    if not separator:
        # The worker baseline uses the same methods with different phrasing.
        before, separator, after = prompt.partition(
            "For known sources/articles resolve/read directly; retain source_id for\nsource-local search and structural/context/range reads."
        )
    if not separator:
        raise ValueError("The reference prompt has no source-local reading instruction")
    return before + _SOURCE_ACTIONS + after


TUNED_COORDINATOR_REFERENCE_PROMPT = (
    _focused_reference(COORDINATOR_REFERENCE_PROMPT) + _METADATA_DELTAS
)
TUNED_RESEARCHER_REFERENCE_PROMPT = (
    _focused_reference(RESEARCHER_REFERENCE_PROMPT) + _METADATA_DELTAS
)

_RELATED_SOURCE_POLICY = """Before a definitive conclusion, examine concrete related leads changing validity, scope
or application. Read their holding, qualifications and dates; distinguish party arguments
and preliminary scope.
Assess bound leads in the SAME answer action's _related_source_reviews: examined for
an operative effect, not_material for a sourced exclusion, unresolved for a precise gap.
source_role describes the original, not favorability; an excluded disposition is operative_text.
Copy lead_id and ranges from navigation and original_evidence_ranges.
State the operative effect and applicability limits with adjacent originals in affected
answer blocks, including a concise explanation for exclusions. Copy those exact sentences
into effect/limitations; summaries must retain relevant qualifications. A surviving general
duty does not establish authority for a disputed sanction or procedural step. Test the exact
violated duty, governing consequence, stage and dates against each original; do not substitute
a related obligation or assume unknown facts. Separate settled effects from supported challenges.
An unread title cannot establish exclusion. If operative text or its decisive interaction
remains unavailable, use submit_partial_answer and disclose that precise gap in its own
uncited paragraph; do not assert that effect as certain.
Reuse delivered originals, repair affected blocks and avoid whole-source rereads, universal
sweeps or deferring a material check to a follow-up.
"""

_before, _separator, _remaining = LEGAL_DEPARTMENT_RESEARCH.partition(
    "related_source_navigation contains"
)
_old_policy, _end, _after = _remaining.partition(
    "Assess the strongest source-supported argument"
)
if not _separator or not _end:
    raise ValueError(
        "The shared legal instruction has no related-source policy section"
    )

_FAVORABLE_APPLICATION = """Assess original-supported favorable grounds and counterarguments; distinguish ordinary
application, an arguable challenge and an established exception. Availability of relief
differs from prior invocation. Apply it to the CURRENT requested outcome, conditionally
when eligibility is unknown; neither another transaction nor a follow-up replaces this
application. State the decisive fact, changed consequence and scope beside the outcome.
Non-invocation or an unknown fact does not prove exclusion; use supplied facts and operative
scope. Never assume fulfillment or success. Keep qualifications in summaries, tables
and detail with adjacent originals. Explain the disputed ground and next step.
"""

_old_application, _application_end, _rest = _after.partition(
    "If authoritative decisions or governing text are unavailable"
)
if not _application_end:
    raise ValueError(
        "The shared legal instruction has no application paragraph boundary"
    )

TUNED_LEGAL_DEPARTMENT_RESEARCH = (
    _before + _RELATED_SOURCE_POLICY + _FAVORABLE_APPLICATION + _application_end + _rest
)
