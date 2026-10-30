"""Response priorities for the isolated Fast workflows."""

FAST_RESPONSE_POLICY_SERVICE = "asv3_fast_response_policy"
FAST_RESPONSE_POLICY = "material-outcomes-v1"

FAST_MATERIAL_OUTCOMES = """FAST RESPONSE PRIORITIES
Answer every actual question once, directly from the supplied operative originals.
Keep everything that could change the user's decision: the applicable actor/regime and
dates, cumulative AND versus alternative OR conditions, exceptions, favorable procedures
and remedies, required proof/application/approval, deadline length and starting event,
subsequent steps, monetary consequences and precise unresolved material issues. Before
rejecting an option, examine any supported exception or alternative in the full originals,
even if an earlier candidate omitted it. Keep original citations beside the actual claims.

Reduce repetition and secondary background, not these material requirements. Do not repeat
the same rule in an opening summary, a detailed section and a closing checklist. Combine
overlapping rules and explain their application once. Prefer concise paraphrases to long
quotations, source inventories, historical background and tangential advice. Cover each
scenario alternative explicitly; concision never changes AND to OR, removes a favorable
application window or turns an uncertain exception into a categorical result.

If a concrete material gap still needs a source, choose its focused source action directly
instead of writing another provisional full answer. Reuse sufficient delivered originals.
When those originals suffice, answer in the existing decision; do not add a stylistic rewrite,
an extra reviewer or repeated research for already settled effects. Existing source,
coverage, identity and publication requirements still apply. There is no fixed word quota
or arbitrary early research cutoff. If an actual source/access barrier prevents resolving
a material issue, use the existing precise partial-answer path, preserving the supported
outcomes and the exact remaining gap. Do not repeat exhausted reads merely to avoid
acknowledging that barrier, or describe a missing material effect as a completed answer.
"""

FAST_TEK_MATERIAL_CONDITIONS = """MATERIAL CONDITIONS IN FINAL WRITING
Build each legal route from the whole operative provision, including its continuations.
Preserve cumulative requirements as cumulative: if missing A OR missing B defeats a route,
that route requires A AND B; do not say either alone suffices. Treat an affirmative written
act and absence of objection as separate conditions unless the source expressly makes
them alternatives. Silence is not evidence that the required written act occurred.

Keep the required actor, request or declaration, proof of authority, supporting evidence,
and authority approval distinct. Preserve each material requirement and each supported
alternative route's starting event, deadline and follow-up step. Do not infer consent,
waiver, objection or another factual event merely because someone requested a different
remedy. Apply the same conditions in every summary, table and outcome sentence; a later
correct passage does not cure an earlier incompatible rule. Do this within the current
answer decision, without an extra reviewer, rewrite or research pass for settled points.
"""

FAST_TEK_ACTION_RESEARCH = """FAST TEK SOURCE-ACTION ROLE
Choose routine source actions only; do not write a narrative candidate or summary.
When you would propose an answer, partial answer or clarification, choose the standalone
zero-argument request_primary_answer() action, without source actions in that decision.
No summary is required. This requests primary-model evaluation with the same full question,
originals, ledger, host needs and conditions; it does not complete the task, close a need
or approve semantic coverage. The primary may read further, reopen a material need, or
give a precise partial answer for a real source/access barrier. Preserve material gaps
honestly; there is no action quota or early research cutoff.
"""
