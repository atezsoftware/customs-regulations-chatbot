import { getRegenerationResearchMode } from "@/sections/chat/ChatUI";

it("preserves Legal Review for a new generation and keeps ASv3 checkpoints in ASv3", () => {
  const input = {
    atezSearchEnabled: false,
    atezSearchV2Enabled: false,
    atezSearchV3Enabled: false,
    legalReviewEnabled: true,
    experimentalResearchEnabled: false,
    experimentalParallelResearchEnabled: false,
    experimentalGuardrailsEnabled: false,
  };
  expect(getRegenerationResearchMode(input)).toMatchObject({
    legalReview: true,
    atezSearchV3: false,
  });
  const resumed = getRegenerationResearchMode({
    ...input,
    asv3ResumeMessageId: 41,
  });
  expect(resumed.atezSearchV3).toBe(true);
  expect(resumed).not.toHaveProperty("legalReview");
});

it("lets the saved ASv3 checkpoint determine its variant when Guardrails is selected", () => {
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
    experimentalGuardrails: false,
    experimentalGuardrailsV2: false,
  });
});

it("keeps Guardrails selected for a new generation without a checkpoint", () => {
  expect(
    getRegenerationResearchMode({
      atezSearchEnabled: false,
      atezSearchV2Enabled: false,
      atezSearchV3Enabled: true,
      experimentalResearchEnabled: false,
      experimentalParallelResearchEnabled: false,
      experimentalGuardrailsEnabled: true,
    })
  ).toMatchObject({
    atezSearchV3: true,
    experimentalGuardrails: true,
    experimentalGuardrailsV2: false,
  });
});

it("lets the saved ASv3 checkpoint determine v2 when Guardrails v2 is selected", () => {
  expect(
    getRegenerationResearchMode({
      asv3ResumeMessageId: 42,
      atezSearchEnabled: false,
      atezSearchV2Enabled: false,
      atezSearchV3Enabled: false,
      experimentalResearchEnabled: false,
      experimentalParallelResearchEnabled: false,
      experimentalGuardrailsEnabled: false,
      experimentalGuardrailsV2Enabled: true,
    })
  ).toEqual({
    atezSearch: false,
    atezSearchV2: false,
    atezSearchV3: true,
    legalComposite: false,
    supersearch: false,
    experimentalResearch: false,
    experimentalParallelResearch: false,
    experimentalGuardrails: false,
    experimentalGuardrailsV2: false,
  });
});

it("keeps Guardrails v2 selected for a new generation without a checkpoint", () => {
  expect(
    getRegenerationResearchMode({
      atezSearchEnabled: false,
      atezSearchV2Enabled: false,
      atezSearchV3Enabled: true,
      experimentalResearchEnabled: false,
      experimentalParallelResearchEnabled: false,
      experimentalGuardrailsEnabled: false,
      experimentalGuardrailsV2Enabled: true,
    })
  ).toMatchObject({
    atezSearchV3: true,
    experimentalGuardrails: false,
    experimentalGuardrailsV2: true,
  });
});

it("keeps Guardrails v3 selected for a new generation without a checkpoint", () => {
  expect(
    getRegenerationResearchMode({
      atezSearchEnabled: false,
      atezSearchV2Enabled: false,
      atezSearchV3Enabled: true,
      experimentalResearchEnabled: false,
      experimentalParallelResearchEnabled: false,
      experimentalGuardrailsEnabled: false,
      experimentalGuardrailsV2Enabled: false,
      experimentalGuardrailsV3Enabled: true,
    })
  ).toMatchObject({
    atezSearchV3: true,
    experimentalGuardrails: false,
    experimentalGuardrailsV2: false,
    experimentalGuardrailsV3: true,
  });
});

it("lets the saved ASv3 checkpoint determine v3 after the input mode changes", () => {
  const mode = getRegenerationResearchMode({
    asv3ResumeMessageId: 43,
    atezSearchEnabled: false,
    atezSearchV2Enabled: false,
    atezSearchV3Enabled: false,
    experimentalResearchEnabled: false,
    experimentalParallelResearchEnabled: false,
    experimentalGuardrailsEnabled: true,
    experimentalGuardrailsV2Enabled: false,
    experimentalGuardrailsV3Enabled: false,
  });
  expect(mode.atezSearchV3).toBe(true);
  expect(mode).not.toHaveProperty("experimentalGuardrailsV3");
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
    experimentalGuardrailsV2: false,
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
  expect(mode.experimentalGuardrailsV2).toBe(false);
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

it("preserves exp cropped regeneration and defers resume variant to its checkpoint", () => {
  const input = {
    atezSearchEnabled: false,
    atezSearchV2Enabled: false,
    atezSearchV3Enabled: false,
    experimentalResearchEnabled: false,
    experimentalParallelResearchEnabled: false,
    experimentalGuardrailsEnabled: false,
    asv3ExpCroppedEnabled: true,
  };
  const fresh = getRegenerationResearchMode(input);
  expect(fresh.asv3ExpCropped).toBe(true);
  expect(fresh).not.toHaveProperty("asv3ExpCheap");
  const resumed = getRegenerationResearchMode({
    ...input,
    asv3ResumeMessageId: 123,
  });
  expect(resumed.atezSearchV3).toBe(true);
  expect(resumed).not.toHaveProperty("asv3ExpCropped");
  expect(resumed).not.toHaveProperty("asv3ExpCheap");
});

it.each([
  ["asv3FastEnabled", "asv3Fast"],
  ["asv3FastCroppedEnabled", "asv3FastCropped"],
] as const)(
  "retains %s for regeneration and leaves resumed identity to the checkpoint",
  (flag, wireFlag) => {
    const input = {
      atezSearchEnabled: false,
      atezSearchV2Enabled: false,
      atezSearchV3Enabled: false,
      experimentalResearchEnabled: false,
      experimentalParallelResearchEnabled: false,
      experimentalGuardrailsEnabled: false,
      [flag]: true,
    };
    expect(getRegenerationResearchMode(input)).toHaveProperty(wireFlag, true);
    const resumed = getRegenerationResearchMode({
      ...input,
      asv3ResumeMessageId: 123,
    });
    expect(resumed.atezSearchV3).toBe(true);
    expect(resumed).not.toHaveProperty("asv3Fast");
    expect(resumed).not.toHaveProperty("asv3FastCropped");
  }
);
