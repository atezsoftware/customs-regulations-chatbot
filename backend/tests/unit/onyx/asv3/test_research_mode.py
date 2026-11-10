from onyx.asv3.research_mode import legislation_research_mode
from onyx.chat.response_workflow import response_workflow
from onyx.server.query_and_chat.models import SendMessageRequest


def test_mode_matches_reserved_response_workflow() -> None:
    single = SendMessageRequest(
        message="Question", atez_search_v3=True, asv3_research_profile="normal"
    )
    parallel = single.model_copy(update={"asv3_exp_cropped": True})
    assert legislation_research_mode(response_workflow(single)) == "single"
    assert legislation_research_mode(response_workflow(parallel)) == "parallel"
    assert legislation_research_mode("asv3_deep") is None
    assert legislation_research_mode(None) is None
