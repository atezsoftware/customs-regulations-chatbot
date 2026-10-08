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
    legalComposite: false,
    supersearch: false,
    experimentalResearch: false,
    experimentalParallelResearch: false,
    experimentalGuardrails: true,
  });
});

it("keeps Legal Composite when regenerating without an ASv3 checkpoint", () => {
  expect(
    getRegenerationResearchMode({
      atezSearchEnabled: false,
      atezSearchV2Enabled: false,
      atezSearchV3Enabled: false,
      legalCompositeEnabled: true,
      experimentalResearchEnabled: false,
      experimentalParallelResearchEnabled: false,
      experimentalGuardrailsEnabled: false,
    })
  ).toEqual({
    atezSearch: false,
    atezSearchV2: false,
    atezSearchV3: false,
    legalComposite: true,
    supersearch: false,
    experimentalResearch: false,
    experimentalParallelResearch: false,
    experimentalGuardrails: false,
  });
});

it("does not reroute an ASv3 checkpoint into currently selected Legal Composite", () => {
  const mode = getRegenerationResearchMode({
    asv3ResumeMessageId: 41,
    atezSearchEnabled: false,
    atezSearchV2Enabled: false,
    atezSearchV3Enabled: false,
    legalCompositeEnabled: true,
    experimentalResearchEnabled: false,
    experimentalParallelResearchEnabled: false,
    experimentalGuardrailsEnabled: false,
  });
  expect(mode.atezSearchV3).toBe(true);
  expect(mode.legalComposite).toBe(false);
  expect(mode.experimentalGuardrails).toBe(false);
});

it("keeps Supersearch for regeneration and preserves an ASv3 checkpoint's workflow", () => {
  const input = {
    atezSearchEnabled: false,
    atezSearchV2Enabled: false,
    atezSearchV3Enabled: false,
    supersearchEnabled: true,
    experimentalResearchEnabled: false,
    experimentalParallelResearchEnabled: false,
    experimentalGuardrailsEnabled: false,
  };
  expect(getRegenerationResearchMode(input)).toMatchObject({
    supersearch: true,
    atezSearchV3: false,
    legalComposite: false,
  });
  expect(
    getRegenerationResearchMode({ ...input, asv3ResumeMessageId: 41 })
  ).toMatchObject({
    supersearch: false,
    atezSearchV3: true,
  });
});
