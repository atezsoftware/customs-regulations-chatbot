import { act, cleanup, renderHook } from "@testing-library/react";
import useDeepResearchToggle from "@/hooks/useDeepResearchToggle";

describe("research mode selection", () => {
  afterEach(cleanup);

  it("selects Legal Composite exclusively and resets on session changes", () => {
    const { result, rerender } = renderHook(
      ({ chatSessionId }) =>
        useDeepResearchToggle({ chatSessionId, agentId: 0 }),
      { initialProps: { chatSessionId: null as string | null } }
    );
    act(() => result.current.toggleAtezSearchV3());
    act(() => result.current.toggleLegalComposite());
    expect(result.current.legalCompositeEnabled).toBe(true);
    expect(result.current.atezSearchV3Enabled).toBe(false);
    expect(result.current.deepResearchEnabled).toBe(false);
    expect(result.current.experimentalResearchEnabled).toBe(false);
    expect(result.current.experimentalParallelResearchEnabled).toBe(false);
    rerender({ chatSessionId: "session-1" });
    expect(result.current.legalCompositeEnabled).toBe(true);
    act(() => result.current.toggleLegalComposite());
    expect(result.current.legalCompositeEnabled).toBe(false);
    act(() => result.current.toggleLegalComposite());
    rerender({ chatSessionId: "session-2" });
    expect(result.current.legalCompositeEnabled).toBe(false);
  });

  it("selects research modes exclusively, and toggles off", () => {
    const { result } = renderHook(() =>
      useDeepResearchToggle({ chatSessionId: null, agentId: 0 })
    );
    expect(result.current.atezSearchV3Enabled).toBe(false);
    expect(result.current.deepResearchEnabled).toBe(false);
    expect(result.current.experimentalResearchEnabled).toBe(false);
    expect(result.current.experimentalParallelResearchEnabled).toBe(false);
    act(() => result.current.toggleAtezSearchV3());
    expect(result.current.atezSearchV3Enabled).toBe(true);
    act(() => result.current.toggleDeepResearch());
    expect(result.current.atezSearchV3Enabled).toBe(false);
    expect(result.current.deepResearchEnabled).toBe(true);
    act(() => result.current.toggleExperimentalResearch());
    expect(result.current.deepResearchEnabled).toBe(false);
    expect(result.current.atezSearchV3Enabled).toBe(false);
    expect(result.current.experimentalResearchEnabled).toBe(true);
    act(() => result.current.toggleExperimentalParallelResearch());
    expect(result.current.experimentalParallelResearchEnabled).toBe(true);
    expect(result.current.experimentalResearchEnabled).toBe(false);
    expect(result.current.deepResearchEnabled).toBe(false);
    expect(result.current.atezSearchV3Enabled).toBe(false);
    act(() => result.current.toggleAtezSearchV3());
    expect(result.current.experimentalParallelResearchEnabled).toBe(false);
    expect(result.current.deepResearchEnabled).toBe(false);
    expect(result.current.experimentalResearchEnabled).toBe(false);
    act(() => result.current.toggleAtezSearchV3());
    expect(result.current.atezSearchV3Enabled).toBe(false);
    expect(result.current.atezSearchEnabled).toBe(false);
    expect(result.current.atezSearchV2Enabled).toBe(false);
    act(() => result.current.toggleExperimentalResearch());
    act(() => result.current.toggleExperimentalResearch());
    expect(result.current.experimentalResearchEnabled).toBe(false);
  });

  it("toggles parallel research off and switches back to nonparallel Experimental", () => {
    const { result } = renderHook(() =>
      useDeepResearchToggle({ chatSessionId: null, agentId: 0 })
    );
    act(() => result.current.toggleExperimentalParallelResearch());
    act(() => result.current.toggleExperimentalParallelResearch());
    expect(result.current.experimentalParallelResearchEnabled).toBe(false);
    act(() => result.current.toggleExperimentalParallelResearch());
    act(() => result.current.toggleExperimentalResearch());
    expect(result.current.experimentalResearchEnabled).toBe(true);
    expect(result.current.experimentalParallelResearchEnabled).toBe(false);
    act(() => result.current.toggleExperimentalParallelResearch());
    act(() => result.current.toggleDeepResearch());
    expect(result.current.deepResearchEnabled).toBe(true);
    expect(result.current.experimentalParallelResearchEnabled).toBe(false);
  });

  it("keeps Experimental Guardrails exclusive with existing ASv3 selections", () => {
    const { result } = renderHook(() =>
      useDeepResearchToggle({ chatSessionId: null, agentId: 0 })
    );

    act(() => result.current.toggleExperimentalParallelResearch());
    expect(result.current.experimentalParallelResearchEnabled).toBe(true);

    act(() => result.current.toggleExperimentalGuardrails());
    expect(result.current.experimentalGuardrailsEnabled).toBe(true);
    expect(result.current.experimentalParallelResearchEnabled).toBe(false);

    act(() => result.current.toggleExperimentalResearch());
    expect(result.current.experimentalGuardrailsEnabled).toBe(false);
    expect(result.current.experimentalResearchEnabled).toBe(true);
  });

  it("switches between Guardrails and Legal Composite exclusively and toggles off", () => {
    const { result } = renderHook(() =>
      useDeepResearchToggle({ chatSessionId: null, agentId: 0 })
    );
    act(() => result.current.toggleExperimentalGuardrails());
    act(() => result.current.toggleLegalComposite());
    expect(result.current.legalCompositeEnabled).toBe(true);
    expect(result.current.experimentalGuardrailsEnabled).toBe(false);
    act(() => result.current.toggleExperimentalGuardrails());
    expect(result.current.experimentalGuardrailsEnabled).toBe(true);
    expect(result.current.legalCompositeEnabled).toBe(false);
    expect(result.current.atezSearchV3Enabled).toBe(false);
    expect(result.current.deepResearchEnabled).toBe(false);
    expect(result.current.experimentalResearchEnabled).toBe(false);
    expect(result.current.experimentalParallelResearchEnabled).toBe(false);
    act(() => result.current.toggleExperimentalGuardrails());
    expect(result.current.experimentalGuardrailsEnabled).toBe(false);
  });

  it("preserves Guardrails on chat creation and resets it on session or agent change", () => {
    const { result, rerender } = renderHook(
      ({ chatSessionId, agentId }) =>
        useDeepResearchToggle({ chatSessionId, agentId }),
      { initialProps: { chatSessionId: null as string | null, agentId: 0 } }
    );
    act(() => result.current.toggleExperimentalGuardrails());
    rerender({ chatSessionId: "session-1", agentId: 0 });
    expect(result.current.experimentalGuardrailsEnabled).toBe(true);
    rerender({ chatSessionId: "session-2", agentId: 0 });
    expect(result.current.experimentalGuardrailsEnabled).toBe(false);
    act(() => result.current.toggleExperimentalGuardrails());
    rerender({ chatSessionId: "session-2", agentId: 1 });
    expect(result.current.experimentalGuardrailsEnabled).toBe(false);
  });

  it("preserves selection for a new session and resets on session or agent switch", () => {
    const { result, rerender } = renderHook(
      ({ chatSessionId, agentId }) =>
        useDeepResearchToggle({ chatSessionId, agentId }),
      { initialProps: { chatSessionId: null as string | null, agentId: 0 } }
    );
    act(() => result.current.toggleAtezSearchV3());
    rerender({ chatSessionId: "session-1", agentId: 0 });
    expect(result.current.atezSearchV3Enabled).toBe(true);
    rerender({ chatSessionId: "session-2", agentId: 0 });
    expect(result.current.atezSearchV3Enabled).toBe(false);
    act(() => result.current.toggleDeepResearch());
    rerender({ chatSessionId: "session-2", agentId: 1 });
    expect(result.current.deepResearchEnabled).toBe(false);
  });

  it("preserves Experimental for a new session and clears it on navigation", () => {
    const { result, rerender } = renderHook(
      ({ chatSessionId, agentId }) =>
        useDeepResearchToggle({ chatSessionId, agentId }),
      { initialProps: { chatSessionId: null as string | null, agentId: 0 } }
    );
    act(() => result.current.toggleExperimentalResearch());
    rerender({ chatSessionId: "session-1", agentId: 0 });
    expect(result.current.experimentalResearchEnabled).toBe(true);
    rerender({ chatSessionId: "session-2", agentId: 0 });
    expect(result.current.experimentalResearchEnabled).toBe(false);
    act(() => result.current.toggleExperimentalResearch());
    rerender({ chatSessionId: "session-2", agentId: 1 });
    expect(result.current.experimentalResearchEnabled).toBe(false);
  });

  it("preserves parallel selection on chat creation and clears it on session or agent changes", () => {
    const { result, rerender } = renderHook(
      ({ chatSessionId, agentId }) =>
        useDeepResearchToggle({ chatSessionId, agentId }),
      { initialProps: { chatSessionId: null as string | null, agentId: 0 } }
    );
    act(() => result.current.toggleExperimentalParallelResearch());
    rerender({ chatSessionId: "session-1", agentId: 0 });
    expect(result.current.experimentalParallelResearchEnabled).toBe(true);
    rerender({ chatSessionId: "session-2", agentId: 0 });
    expect(result.current.experimentalParallelResearchEnabled).toBe(false);
    act(() => result.current.toggleExperimentalParallelResearch());
    rerender({ chatSessionId: "session-2", agentId: 1 });
    expect(result.current.experimentalParallelResearchEnabled).toBe(false);
  });
});
