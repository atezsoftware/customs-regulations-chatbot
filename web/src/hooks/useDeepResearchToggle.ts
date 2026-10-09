"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import type { WorkflowSelection } from "@/lib/chat/interfaces";

interface UseDeepResearchToggleProps {
  chatSessionId: string | null;
  agentId: number | undefined;
}

/** Keep research selection exclusive and reset it on session or agent changes. */
export default function useDeepResearchToggle({
  chatSessionId,
  agentId,
}: UseDeepResearchToggleProps) {
  const [mode, setMode] = useState<WorkflowSelection | null>(null);
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
  const toggleExperimentalResearch = useCallback(() => {
    setMode((current) => (current === "experimental" ? null : "experimental"));
  }, []);
  const toggleExperimentalParallelResearch = useCallback(() => {
    setMode((current) =>
      current === "experimental_parallel" ? null : "experimental_parallel"
    );
  }, []);
  const toggleExperimentalGuardrails = useCallback(() => {
    setMode((current) =>
      current === "experimental_guardrails" ? null : "experimental_guardrails"
    );
  }, []);
  const toggleExperimentalGuardrailsV2 = useCallback(() => {
    setMode((current) =>
      current === "experimental_guardrails_v2"
        ? null
        : "experimental_guardrails_v2"
    );
  }, []);
  const toggleExperimentalGuardrailsV3 = useCallback(() => {
    setMode((current) =>
      current === "experimental_guardrails_v3"
        ? null
        : "experimental_guardrails_v3"
    );
  }, []);

  const toggleLegalComposite = useCallback(() => {
    setMode((current) =>
      current === "legal_composite" ? null : "legal_composite"
    );
  }, []);

  const toggleSupersearch = useCallback(() => {
    setMode((current) => (current === "supersearch" ? null : "supersearch"));
  }, []);

  return {
    supersearchEnabled: mode === "supersearch",
    toggleSupersearch,
    legalCompositeEnabled: mode === "legal_composite",
    toggleLegalComposite,
    deepResearchEnabled: mode === "deep",
    toggleDeepResearch,
    atezSearchV3Enabled: mode === "normal",
    toggleAtezSearchV3,
    experimentalResearchEnabled: mode === "experimental",
    toggleExperimentalResearch,
    experimentalParallelResearchEnabled: mode === "experimental_parallel",
    toggleExperimentalParallelResearch,
    experimentalGuardrailsEnabled: mode === "experimental_guardrails",
    toggleExperimentalGuardrails,
    experimentalGuardrailsV2Enabled: mode === "experimental_guardrails_v2",
    toggleExperimentalGuardrailsV2,
    experimentalGuardrailsV3Enabled: mode === "experimental_guardrails_v3",
    toggleExperimentalGuardrailsV3,
    atezSearchEnabled: false,
    atezSearchV2Enabled: false,
    toggleAtezSearch: undefined,
    toggleAtezSearchV2: undefined,
  };
}
