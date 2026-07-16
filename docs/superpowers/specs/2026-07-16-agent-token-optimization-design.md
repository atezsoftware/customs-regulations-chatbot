# Agent Token Optimization Design

## Goal

Reduce per-question total tokens without reducing retrieval coverage, factual accuracy, citation quality, resumability, or final-answer quality.

## Baseline and acceptance gates

The benchmark uses a fixed synthetic customs corpus and questions covering a direct rule, an exception, and a cross-document reference. It records action calls, summarization calls, serialized prompt payload, input/output/thinking tokens when a real Gemini client is available, required facts, and required citations. Every optimization must keep all deterministic accuracy assertions passing. The final comparison runs the same benchmark against the pre-optimization commit and the optimized tree.

## Changes

1. Split the repeated prompt into a compact action prompt and a final-answer prompt. Action and history-compaction calls use low thinking; the final synthesis keeps high thinking.
2. Replace fixed-turn history retention with a token-budgeted compactor. Preserve the original task, a compact evidence summary, and the newest action/tool exchange. Avoid re-summarizing unchanged history.
3. Add `get_chunk_context`, which expands a semantic hit to a bounded neighboring chunk window. Keep `get_document` as an explicit fallback when broader context is genuinely required.
4. Add a bounded batch action containing at most three independent tool calls. Execute the calls together and return a single combined result turn, reducing planner round trips without reducing retrieval breadth.

## Safety

The max-step safety net is not lowered. Full-document retrieval remains available. Batch size and chunk windows are bounded. Existing Continue/resume, duplicate-query, citation, Unicode, and stream behavior remain covered by regression tests.
