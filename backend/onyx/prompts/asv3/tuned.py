"""Source-lead follow-through for the isolated normal-profile experiment."""

from onyx.prompts.asv3.research import LEGAL_DEPARTMENT_RESEARCH

TUNED_PROMPT_VERSION = "asv3-tuned-2026-10-06.1"

_RELATED_SOURCE_POLICY = """Before a definitive legal application, check whether available originals or related
source leads identify a judgment, amendment or other authority that could change the rule's
scope, validity or application. The user need not already know that the issue is disputed.
For concrete leads attached to delivered governing originals, read the candidate's operative
holding, connected qualifications and applicable dates. Distinguish the disposition from
party arguments, and ordinary administrative practice from a source-supported challenge.
Use the SAME answer action's _related_source_reviews to assess those leads: examined for an
operative effect, not_material for a reasoned exclusion based on that candidate's own text,
or unresolved for a precise missing interaction. Copy lead_id and original witness ranges
from the supplied navigation and original_evidence_ranges; never invent citation numbers
or offsets. Acquired passages are not automatically an examined holding.
Retain each material examined effect and its limitations with the candidate's own adjacent
original citations. Copy effect and limitations into the review as exact sentences from those
answer blocks; do not rewrite other answer blocks to close metadata. An unread lead cannot
be dismissed from its title. If its
operative text cannot be obtained, use submit_partial_answer and disclose that exact gap
in a separate uncited paragraph; do not still assert the unresolved effect as certain.
Reuse sufficient delivered originals. No universal court sweep, whole-source rereading,
separate reviewer call or deferral of this material check to an optional follow-up.
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
