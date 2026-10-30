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
  const toggleAsv3ExpCropped = useCallback(() => {
    setMode((current) =>
      current === "asv3_exp_cropped" ? null : "asv3_exp_cropped"
    );
  }, []);
  const toggleAsv3Fast = useCallback(() => {
    setMode((current) => (current === "asv3_fast" ? null : "asv3_fast"));
  }, []);
  const toggleAsv3FastCropped = useCallback(() => {
    setMode((current) =>
      current === "asv3_fast_cropped" ? null : "asv3_fast_cropped"
    );
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

  const toggleLegalReview = useCallback(() => {
    setMode((current) => (current === "legal_review" ? null : "legal_review"));
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
    legalReviewEnabled: mode === "legal_review",
    toggleLegalReview,
    supersearchEnabled: mode === "supersearch",
    toggleSupersearch,
    legalCompositeEnabled: mode === "legal_composite",
    toggleLegalComposite,
    deepResearchEnabled: mode === "deep",
    toggleDeepResearch,
    atezSearchV3Enabled: mode === "normal",
    toggleAtezSearchV3,
    asv3ExpCroppedEnabled: mode === "asv3_exp_cropped",
    toggleAsv3ExpCropped,
    asv3FastEnabled: mode === "asv3_fast",
    toggleAsv3Fast,
    asv3FastCroppedEnabled: mode === "asv3_fast_cropped",
    toggleAsv3FastCropped,
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
