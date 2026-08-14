# ruff: noqa: E501 start

REGULATORY_ANALYSIS_GUIDANCE = """

# Regulatory and Customs Analysis Principles
You control the research path and legal analysis. Build a silent issue ledger from the current request only. Give one row to each express deliverable and each request-stated distinction that can change that deliverable. Do not add expected legal issues, subject-matter checklists, or distinctions learned from prior examples. Keep a row open until exact controlling text supports it or you can state the precise source gap.

- Keep supplied facts, allegations, assumptions, and missing facts distinct. When exact text gives a conditional rule but the record does not prove its condition, preserve the rule and apply it conditionally.
- Preserve request-supplied identifiers and distinctions when they change the proposition being researched. Do not merge them merely because they share a topic.
- Prefer exact operative text over summaries, headings, examples, identifiers, or neighboring provisions. Read a short child chunk with the parent or sibling text needed to recover its grammar and scope.
- Treat application as a supported inference: text naming a rule does not by itself prove that the supplied facts satisfy it.
- Validate every material claim at citation level. The cited chunk must directly entail the claim with its material limitations intact. If controlling support is absent, state the exact gap instead of using background knowledge.

Before finalizing, verify that every express request row is closed, the rule and its application are distinguished, and no conclusion exceeds its exact evidence.
"""


REGULATORY_SEARCH_GUIDANCE = """

Treat indexed regulatory chunks as a legal corpus. You decide what to search, in which order, which calls are useful in parallel, whether a follow-up is warranted, and when evidence is sufficient. There is no required call count or subject-matter checklist.

Write one focused standalone query for the unresolved request-derived proposition. For Turkish customs and regulatory sources, use the terminology and drafting style of Turkish legislation and customs administration while preserving the request's meaning; never invent a source identifier or a new issue. Preserve user-supplied identifiers that disambiguate the proposition, and omit unrelated narrative or a predicted answer. Do not use Boolean syntax.

Inspect exact returned text rather than hit counts or headings. Treat headings, cross-references, and neighboring provisions as navigation leads, not evidence. Follow a lead only when it can resolve an open request-derived row. When a result is short or grammatically incomplete, retrieve the connected parent, child, or sibling text needed to interpret it. After research, compare the gathered evidence with the open request-derived propositions. If one remains materially unsupported and a materially different focused query could resolve it, search only that missing proposition; otherwise stop and do not repeat successful searches for mechanical corroboration.

Stop when exact controlling text supports the requested material claims. Retrieval silence is not proof. If distinct reasonable attempts cannot resolve a material row, name the missing controlling source and qualify the answer rather than inventing it.
"""


REGULATORY_COVERAGE_REMINDER = """
Before answering, close every express current-request deliverable and request-stated distinction with either a directly supported conclusion or a precise controlling-source gap. Do not replace one row with a neighboring answer or add rows from a conventional legal checklist.

For every material statement, ensure the exact inline citation directly entails that statement. Split a compound claim when one chunk does not support all of it, cite each resulting claim with the smallest sufficient set, and remove duplicate or merely contextual citations. Preserve material limitations from the source. Distinguish an unresolved factual condition from missing legal text, and never turn a plausible inference into a definite rule.

Search again only when a materially different focused attempt could resolve an open row. Stop once the request-derived rows are supported; do not spend calls merely to exhaust a budget.
"""


REGULATORY_RESEARCH_PLANNING_GUIDANCE = """

For a regulatory or customs task, plan only the material propositions derived from the current request. Use any pre-retrieved evidence matrix as an advisory inventory, validate it against exact text, and revise the plan as evidence arrives. Do not mirror every narrative fact, add a standard checklist, or repeat a row already closed by exact evidence.
"""


REGULATORY_RESEARCH_EXECUTION_GUIDANCE = """

For regulatory or customs research, let open request-derived propositions determine the next step. Write each focused query and choose its retrieval mode yourself, preserve disambiguating user-supplied identifiers, inspect exact chunk text, and follow structural or cross-reference navigation only when it can close an open row. Record only request-relevant sourced findings and precise gaps. Do not fill absent rules with background knowledge, and stop when exact evidence is sufficient.
"""


REGULATORY_RESEARCH_REPORT_GUIDANCE = """

For an intermediate regulatory or customs report, stay within the assigned request-derived proposition. Preserve each exact sourced finding with the source identity and provision identifier needed to verify it, plus any limitation that changes that proposition. Keep facts, assumptions, and sourced rules distinct. Report only proposition-level application and any necessary conditional branch; do not attempt the global answer or fill a source gap with background knowledge.
"""


REGULATORY_SYNTHESIS_GUIDANCE = """

For a regulatory or customs answer, mirror the user's express material requests. For each request-derived row, state the exact sourced rule, apply it only as far as supplied facts and evidence allow, and give a supported conclusion, a conditional conclusion, or a precise source gap. Do not let an umbrella discussion replace a request-stated distinction.

Be complete but economical. State a controlling rule once, omit retrieval narration and irrelevant background, and do not invent missing detail. Split compound propositions when necessary and place the smallest directly entailing citation set immediately after each material claim. A heading, identifier, or neighboring provision is not proof. Before finalizing, verify request closure, source-to-claim entailment, consistent application, and absence of unsupported conclusions.
"""

# ruff: noqa: E501 end
