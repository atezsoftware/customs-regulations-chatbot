"use client";

import { useState, useEffect, useRef, useCallback } from "react";

interface UseDeepResearchToggleProps {
  chatSessionId: string | null;
  agentId: number | undefined;
}

/**
 * Custom hook for managing the agent search (deep research) toggle state.
 * Automatically resets the toggle to false when:
 * - Switching between existing chat sessions
 * - The assistant changes
 * - The page is reloaded (since state initializes to false)
 *
 * The toggle is preserved when transitioning from no chat session to a new session.
 *
 * @param chatSessionId - The current chat session ID
 * @param agentId - The current agent ID
 * @returns An object containing the toggle state and toggle function
 */
export default function useDeepResearchToggle({
  chatSessionId,
  agentId,
}: UseDeepResearchToggleProps) {
  const [deepResearchEnabled, setDeepResearchEnabled] = useState(false);
  const [atezSearchEnabled, setAtezSearchEnabled] = useState(false);
  const [atezSearchV2Enabled, setAtezSearchV2Enabled] = useState(false);
  const [atezSearchV3Enabled, setAtezSearchV3Enabled] = useState(false);
  const previousChatSessionId = useRef<string | null>(chatSessionId);

  // Reset when switching chat sessions, but preserve when going from null to a new session
  useEffect(() => {
    const previousId = previousChatSessionId.current;
    previousChatSessionId.current = chatSessionId;

    // Only reset if we're switching between actual sessions (not from null to a new session)
    if (previousId !== null && previousId !== chatSessionId) {
      setDeepResearchEnabled(false);
      setAtezSearchEnabled(false);
      setAtezSearchV2Enabled(false);
      setAtezSearchV3Enabled(false);
    }
  }, [chatSessionId]);

  // Reset when switching assistants
  useEffect(() => {
    setDeepResearchEnabled(false);
    setAtezSearchEnabled(false);
    setAtezSearchV2Enabled(false);
    setAtezSearchV3Enabled(false);
  }, [agentId]);

  const toggleDeepResearch = useCallback(() => {
    setDeepResearchEnabled((enabled) => {
      if (!enabled) setAtezSearchV3Enabled(false);
      return !enabled;
    });
  }, []);

  const toggleAtezSearch = useCallback(() => {
    setAtezSearchEnabled((enabled) => {
      const next = !enabled;
      if (next) {
        setAtezSearchV2Enabled(false);
        setAtezSearchV3Enabled(false);
      }
      return next;
    });
  }, []);

  const toggleAtezSearchV2 = useCallback(() => {
    setAtezSearchV2Enabled((enabled) => {
      const next = !enabled;
      if (next) {
        setAtezSearchEnabled(false);
        setAtezSearchV3Enabled(false);
      }
      return next;
    });
  }, []);

  const toggleAtezSearchV3 = useCallback(() => {
    setAtezSearchV3Enabled((enabled) => {
      if (!enabled) {
        setAtezSearchEnabled(false);
        setAtezSearchV2Enabled(false);
        setDeepResearchEnabled(false);
      }
      return !enabled;
    });
  }, []);

  return {
    deepResearchEnabled,
    toggleDeepResearch,
    atezSearchEnabled,
    toggleAtezSearch,
    atezSearchV2Enabled,
    toggleAtezSearchV2,
    atezSearchV3Enabled,
    toggleAtezSearchV3,
  };
}
