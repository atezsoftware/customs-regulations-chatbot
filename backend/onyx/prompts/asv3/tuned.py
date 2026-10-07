"""Source-lead follow-through for the isolated normal-profile experiment."""

from onyx.prompts.asv3.coordinator_reference import (
    COORDINATOR_REFERENCE_PROMPT,
    RESEARCHER_REFERENCE_PROMPT,
)
from onyx.prompts.asv3.research import LEGAL_DEPARTMENT_RESEARCH

TUNED_PROMPT_VERSION = "asv3-tuned-2026-10-07.2"

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

_RELATED_SOURCE_POLICY = """Before a definitive conclusion, check originals and concrete related-source leads for
authorities changing the rule's validity, scope or application. The user need not know the dispute.
For concrete leads attached to delivered governing originals, read the candidate's operative
holding, connected qualifications and applicable dates. Distinguish the disposition from
party arguments, and ordinary administrative practice from a source-supported challenge.
An application subject, requested relief or procedural introduction is not a disposition;
locate the candidate's own holding and connected qualifications, not its whole file.
Use the SAME answer action's _related_source_reviews to assess those leads: examined for an
operative effect, not_material for a reasoned exclusion based on that candidate's own text,
or unresolved for a precise missing interaction. Copy lead_id and witness ranges from
navigation and original_evidence_ranges; never invent citations or offsets.
Apply each material examined effect and its limitations to every affected conclusion,
including summaries, with its own adjacent original citations. Copy effect and limitations
into the review as exact sentences from those
answer blocks; do not rewrite other blocks to close metadata. Do not dismiss an unread lead
from its title. If its
operative text cannot be obtained, use submit_partial_answer and disclose that exact gap
in a separate uncited paragraph; do not still assert the unresolved effect as certain.
Reuse sufficient delivered originals. No universal court sweep, whole-source rereading,
or deferral of this material check to an optional follow-up.
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

TUNED_LEGAL_DEPARTMENT_RESEARCH = (
    _before
    + _RELATED_SOURCE_POLICY
    + "Assess the strongest source-supported argument"
    + _after
)
