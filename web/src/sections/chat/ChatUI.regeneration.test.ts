import { getRegenerationResearchMode } from "@/sections/chat/ChatUI";

it("preserves Experimental Guardrails when regenerating from its checkpoint", () => {
  expect(
    getRegenerationResearchMode({
      asv3ResumeMessageId: 41,
      atezSearchEnabled: false,
      atezSearchV2Enabled: false,
      atezSearchV3Enabled: false,
      experimentalResearchEnabled: false,
      experimentalParallelResearchEnabled: false,
      experimentalGuardrailsEnabled: true,
    })
  ).toEqual({
    atezSearch: false,
    atezSearchV2: false,
    atezSearchV3: true,
    experimentalResearch: false,
    experimentalParallelResearch: false,
    experimentalGuardrails: true,
  });
});
