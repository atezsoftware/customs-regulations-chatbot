import { act, renderHook } from "@testing-library/react";
import useDeepResearchToggle from "@/hooks/useDeepResearchToggle";

describe("research workflow selection", () => {
  it("starts in normal mode and preserves deep research when the session is created", () => {
    const { result, rerender } = renderHook(
      ({ chatSessionId, agentId }) =>
        useDeepResearchToggle({ chatSessionId, agentId }),
      { initialProps: { chatSessionId: null as string | null, agentId: 0 } }
    );
    expect(result.current.deepResearchEnabled).toBe(false);
    act(() => result.current.toggleDeepResearch());
    rerender({ chatSessionId: "new-session", agentId: 0 });
    expect(result.current.deepResearchEnabled).toBe(true);
    expect(result.current.atezSearchV3Enabled).toBe(false);
    rerender({ chatSessionId: "another-session", agentId: 0 });
    expect(result.current.deepResearchEnabled).toBe(false);
    act(() => result.current.toggleDeepResearch());
    rerender({ chatSessionId: "another-session", agentId: 1 });
    expect(result.current.deepResearchEnabled).toBe(false);
  });
});
