"""Names the research workflow that produced an answer.

Recorded on the assistant message so the workflow can be shown under the answer
after the chat is reopened; the request flags themselves are not stored.
Identifiers are stable values; the UI maps them to the labels on the input bar.
"""

from onyx.server.query_and_chat.models import SendMessageRequest


def _asv3_workflow(request: SendMessageRequest) -> str:
    # Variant flags override the profile they run under, so they are checked first.
    if request.asv3_fast:
        return "asv3_fast"
    if request.asv3_fast_cropped:
        return "asv3_fast_cropped"
    if request.asv3_exp_cropped:
        return "asv3_exp_cropped"
    if request.asv3_guardrails_v3:
        return "experimental_guardrails_v3"
    if request.asv3_guardrails_v2:
        return "experimental_guardrails_v2"
    if request.asv3_guarded_experimental:
        return "experimental_guardrails"
    if request.asv3_research_profile == "experimental":
        return "experimental_asv3" if request.asv3_parallel_research else "experimental"
    if request.asv3_research_profile == "normal":
        return "asv3"
    return "asv3_deep"


def response_workflow(request: SendMessageRequest) -> str | None:
    if request.legal_review:
        return "legal_review"
    if request.supersearch:
        return "supersearch"
    if request.legal_composite:
        return "legal_composite"
    if request.deep_research:
        return "deep_research"
    if request.atez_search_v3:
        return _asv3_workflow(request)
    if request.atez_search_v2:
        return "asv2"
    if request.atez_search:
        return "asv1"
    return None
