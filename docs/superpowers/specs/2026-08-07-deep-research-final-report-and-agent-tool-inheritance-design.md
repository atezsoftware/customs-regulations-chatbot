# Deep Research Final Report Reliability and Agent Tool Inheritance

## Issues to Address

1. Gemini 3.x can spend the final-report output allowance on thinking tokens and
   finish with no user-visible answer. `generate_final_report` currently turns
   that condition into the generic `LLM failed to generate the final deep
   research report` `ValueError`.
2. Custom agents do not automatically receive the tools enabled for the default
   chat assistant.
3. Deep research receives a custom-agent prompt argument but discards it in its
   clarification, planning, orchestration, and final-report histories.

## Important Notes

- The failing default-assistant and Customs Agent turns reached different
  regulatory/non-regulatory branches but failed at the same final synthesis
  boundary.
- Gemini 3 Flash models cannot fully disable thinking. Omitting a thinking
  setting selects the model default, which is medium for the Flash family.
- Thinking tokens count against output usage. The current final-report ceilings
  are 20,000 tokens for general research and 8,192 tokens for regulatory
  research.
- Final-report synthesis follows completed planning and research. Capping only
  this synthesis pass at low thinking preserves the reasoning quality of all
  preceding research stages while reserving output capacity for the report.
- A partial or mechanically concatenated research-note fallback is not an
  acceptable final report. Reliability must not lower answer quality.
- Tool inheritance must be enforced by the backend. A frontend-only union would
  advertise permissions that the server does not authorize.

## Design

### Final-report reasoning budget

Every deep-research final synthesis attempt uses at most `LOW` reasoning,
regardless of a higher chat-level reasoning selection. Planning, orchestration,
research-agent work, evidence review, and correction retain the user's selected
reasoning level.

The initial report is staged in a `BufferedEmitter` and an isolated
`ChatStateContainer`. If it produces a usable answer, the staged packets and
state are committed exactly once. If it ends without an answer, the failure is
logged with provider, model, finish reason, reasoning presence, raw-answer
presence, and attempt number.

For retryable empty results, including a token-limit finish, the report is
generated one more time at `OFF`. Provider request shaping must translate `OFF`
correctly: Gemini 3 Flash receives its supported minimal thinking level rather
than an omitted setting that silently restores medium thinking. Other provider
semantics remain provider-appropriate.

Only a complete, usable report is published. If both synthesis attempts fail,
the existing empty-response classifier raises a typed error containing the real
finish reason. The generic final-report `ValueError` is removed.

### Agent tool inheritance

The effective runtime tool set for every non-default agent is:

`default Assistant tools union agent-specific tools`

Tools are deduplicated by database tool ID. The default Assistant remains the
admin-managed baseline, so enabling or disabling a normal-chat tool takes
effect for all custom agents without copying relationship rows.

Runtime authorization, tool construction, and user-facing agent snapshots use
the effective set. Agent create/edit persistence continues to use only directly
assigned tools so inherited tools are not materialized into every agent record.
Existing per-user disabled-tool preferences continue to filter the effective
set at request time.

### Custom-agent deep research

The custom-agent prompt is inserted after the deep-research system prompt in
clarification, plan generation, orchestration, and final synthesis histories.
The deep-research execution allowlist remains internal search only; inherited
non-research tools remain available in normal agent chat. Because Internal
Search is inherited from the default Assistant, every custom agent receives the
Deep Research control whenever the workspace-level feature is enabled.

## Error Handling and Observability

- Empty final synthesis is classified using `LlmStepResult.finish_reason`
  instead of a generic `ValueError`.
- Each failed staged attempt emits one structured warning with model/provider,
  finish reason, and whether reasoning or raw text was received.
- Discarded attempts never emit partial answer, reasoning, citation, or timing
  packets to the client.
- Provider transport, authentication, and quota failures keep their existing
  typed error paths and are not hidden behind retries intended for empty output.

## Tests

### Unit tests

- A final report requested at high/auto reasoning is sent at low reasoning.
- A length-terminated empty first attempt retries once at off/minimal reasoning
  and publishes only the successful second report.
- Two unusable attempts raise a classified empty-response error with the
  terminal finish reason, never the old generic `ValueError`.
- Regulatory and non-regulatory report paths share the staged retry behavior.
- Gemini `ReasoningEffort.OFF` is sent to LiteLLM as `none`, allowing LiteLLM to
  map Gemini 3 Flash to minimal thinking.
- Custom-agent prompt text appears in all deep-research stage histories.
- Effective agent tools contain the default baseline plus agent-specific tools,
  without duplicates, and allowed-tool filtering still applies.

### Integration coverage

- A custom-agent chat exposes the same admin-enabled baseline tools as default
  chat and offers Deep Research when Internal Search is enabled.
- A live Gemini deep-research turn verifies that final synthesis produces a
  report without the former generic error.

## Non-goals

- Increasing the number of research-agent cycles or reducing retrieved evidence.
- Publishing partial research notes as a final report.
- Allowing image generation, code execution, or arbitrary MCP actions inside
  the deep-research orchestrator.
- Copying default tool relationships into each custom-agent database row.
