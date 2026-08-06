from onyx.configs.constants import MessageType
from onyx.llm.interfaces import LLM
from onyx.secondary_llm_flows.query_expansion import (
    keyword_query_expansion,
    semantic_query_rephrase,
)
from onyx.tools.models import ChatMinimalTextMessage
from onyx.utils.threadpool_concurrency import run_functions_tuples_in_parallel


def expand_search_queries(
    user_query: str,
    llm: LLM,
) -> tuple[str | None, list[str]]:
    """Generate the shared semantic and legal/lexical Search UI variants."""
    history = [
        ChatMinimalTextMessage(
            message=user_query,
            message_type=MessageType.USER,
        )
    ]
    semantic_result, keyword_result = run_functions_tuples_in_parallel(
        [
            (semantic_query_rephrase, (history, llm)),
            (keyword_query_expansion, (history, llm)),
        ],
        allow_failures=True,
    )

    semantic_query = (
        semantic_result.strip()
        if isinstance(semantic_result, str) and semantic_result.strip()
        else None
    )
    keyword_queries = (
        [query.strip() for query in keyword_result if query.strip()]
        if isinstance(keyword_result, list)
        and all(isinstance(query, str) for query in keyword_result)
        else []
    )
    return semantic_query, keyword_queries
