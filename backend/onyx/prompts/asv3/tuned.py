"""Source-lead follow-through for the isolated normal-profile experiment."""

from onyx.prompts.asv3.coordinator_reference import (
    COORDINATOR_REFERENCE_PROMPT,
    RESEARCHER_REFERENCE_PROMPT,
)
from onyx.prompts.asv3.research import LEGAL_DEPARTMENT_RESEARCH

TUNED_PROMPT_VERSION = "asv3-tuned-2026-10-07.14"

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

_RELATED_SOURCE_POLICY = """Before a definitive conclusion, examine concrete related-source leads changing the rule's
validity, scope or application. The user need not know the dispute. Read the candidate's
operative holding, connected qualifications and dates. Distinguish these from party arguments,
requested relief and preliminary scope; separate administrative practice from a sourced challenge.
Use the SAME answer action's _related_source_reviews to assess those leads: examined for an
operative effect, not_material for a reasoned exclusion based on that candidate's own text,
or unresolved for a precise missing interaction. source_role describes the selected original,
not whether it favors this case: a disposition remains operative_text even if not_material.
Copy lead_id and ranges from navigation and original_evidence_ranges.
Apply each material examined effect and its limitations to every affected conclusion,
including summaries, with its own adjacent original citations. Copy effect and limitations
into the review as exact sentences from those answer blocks. Preserve other blocks when
repairing metadata. An unread title does not establish exclusion. If its
operative text cannot be obtained, use submit_partial_answer and disclose that exact gap
in a separate uncited paragraph; do not still assert the unresolved effect as certain.
Reuse delivered originals; avoid universal court sweeps, whole-source rereading and deferring
a material check to an optional follow-up.
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

_FAVORABLE_APPLICATION = """Assess the strongest original-supported favorable ground and counterargument; distinguish
ordinary application, an arguable challenge and an established exception. For relief,
remedies or alternatives, availability differs from prior invocation. Non-invocation or
an unknown eligibility fact does not exclude the option: explain its relevant conditional
branch, decisive fact and changed consequence beside each affected outcome. Exclusion needs
supplied facts and the original's actual scope. Never assume fulfillment, a factual bar or
success. Keep these conditions and uncertainty in summaries, tables and detail with adjacent
originals. Explain the specific disputed ground and next step.
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
