import { act, cleanup, renderHook } from "@testing-library/react";
import useDeepResearchToggle from "@/hooks/useDeepResearchToggle";

describe("useDeepResearchToggle", () => {
  afterEach(cleanup);

  it("manages Atez Search independently from Deep Research", () => {
    const { result } = renderHook(() =>
      useDeepResearchToggle({ chatSessionId: null, agentId: 0 })
    );

    act(() => result.current.toggleAtezSearch());
    expect(result.current.atezSearchEnabled).toBe(true);
    expect(result.current.deepResearchEnabled).toBe(false);

    act(() => result.current.toggleDeepResearch());
    expect(result.current.atezSearchEnabled).toBe(true);
    expect(result.current.deepResearchEnabled).toBe(true);
  });

  it("resets both research modifiers when switching existing sessions", () => {
    const { result, rerender } = renderHook(
      ({ chatSessionId }) =>
        useDeepResearchToggle({ chatSessionId, agentId: 0 }),
      { initialProps: { chatSessionId: "session-1" as string | null } }
    );

    act(() => {
      result.current.toggleAtezSearch();
      result.current.toggleDeepResearch();
    });
    rerender({ chatSessionId: "session-2" });

    expect(result.current.atezSearchEnabled).toBe(false);
    expect(result.current.deepResearchEnabled).toBe(false);
  });

  it("keeps Atez Search versions mutually exclusive", () => {
    const { result } = renderHook(() =>
      useDeepResearchToggle({ chatSessionId: null, agentId: 0 })
    );

    act(() => result.current.toggleAtezSearch());
    act(() => result.current.toggleAtezSearchV2());
    expect(result.current.atezSearchEnabled).toBe(false);
    expect(result.current.atezSearchV2Enabled).toBe(true);

    act(() => result.current.toggleAtezSearch());
    expect(result.current.atezSearchEnabled).toBe(true);
    expect(result.current.atezSearchV2Enabled).toBe(false);
  });
});
