"use client";

import { useState, useEffect, useRef, useCallback } from "react";

interface UseDeepResearchToggleProps {
  chatSessionId: string | null;
  agentId: number | undefined;
}

/** Keep research selection exclusive and reset it on session or agent changes. */
export default function useDeepResearchToggle({
  chatSessionId,
  agentId,
}: UseDeepResearchToggleProps) {
  const [mode, setMode] = useState<"normal" | "deep" | null>(null);
  const previousChatSessionId = useRef<string | null>(chatSessionId);

  useEffect(() => {
    const previousId = previousChatSessionId.current;
    previousChatSessionId.current = chatSessionId;
    if (previousId !== null && previousId !== chatSessionId) setMode(null);
  }, [chatSessionId]);

  useEffect(() => setMode(null), [agentId]);

  const toggleDeepResearch = useCallback(() => {
    setMode((current) => (current === "deep" ? null : "deep"));
  }, []);
  const toggleAtezSearchV3 = useCallback(() => {
    setMode((current) => (current === "normal" ? null : "normal"));
  }, []);

  return {
    deepResearchEnabled: mode === "deep",
    toggleDeepResearch,
    atezSearchV3Enabled: mode === "normal",
    toggleAtezSearchV3,
    atezSearchEnabled: false,
    atezSearchV2Enabled: false,
    toggleAtezSearch: undefined,
    toggleAtezSearchV2: undefined,
  };
}
