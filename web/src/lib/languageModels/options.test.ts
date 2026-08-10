import { buildLlmOptions, llmOptionKey } from "@/lib/languageModels/options";
import { structureValue } from "@/lib/languageModels/utils";
import type {
  LLMProviderDescriptor,
  ModelConfiguration,
} from "@/lib/languageModels/types";

function makeModelConfiguration(id: number, name: string): ModelConfiguration {
  return {
    id,
    name,
    is_visible: true,
    max_input_tokens: null,
    supports_image_input: false,
    supports_reasoning: false,
    effectiveDisplayName: name,
  };
}

function makeProvider(
  id: number,
  name: string | null,
  provider: string,
  modelConfigurations: ModelConfiguration[]
): LLMProviderDescriptor {
  return {
    id,
    name,
    provider,
    provider_display_name: name ?? provider,
    model_configurations: modelConfigurations,
  };
}

describe("llmOptionKey", () => {
  it("gives distinct keys to same-named models from different providers", () => {
    const providers = [
      makeProvider(1, "OpenAI Main", "openai", [
        makeModelConfiguration(11, "gpt-4o"),
      ]),
      makeProvider(2, "OpenAI Backup", "openai", [
        makeModelConfiguration(22, "gpt-4o"),
      ]),
    ];

    const keys = buildLlmOptions(providers).map(llmOptionKey);

    expect(keys).toHaveLength(2);
    expect(new Set(keys).size).toBe(2);
  });

  it("keys by model configuration id when present", () => {
    expect(
      llmOptionKey({
        provider: "openai",
        modelName: "gpt-4o",
        modelConfigurationId: 11,
      })
    ).toBe("mc:11");
  });

  it("falls back to provider + model name without an id", () => {
    expect(
      llmOptionKey({
        provider: "openai",
        modelName: "gpt-4o",
        modelConfigurationId: null,
      })
    ).toBe("openai:gpt-4o");
    expect(llmOptionKey({ provider: "openai", modelName: "gpt-4o" })).toBe(
      "openai:gpt-4o"
    );
  });
});

describe("buildLlmOptions", () => {
  it("uses the provider type as the selector for a nameless provider", () => {
    const providers = [
      makeProvider(1, null, "vertex_ai", [
        makeModelConfiguration(11, "gemini-3.6-flash"),
      ]),
    ];

    expect(buildLlmOptions(providers)).toEqual([
      expect.objectContaining({
        name: "vertex_ai",
        provider: "vertex_ai",
        modelName: "gemini-3.6-flash",
      }),
    ]);

    const option = buildLlmOptions(providers)[0]!;
    expect(structureValue(option.name, option.provider, option.modelName)).toBe(
      "vertex_ai__vertex_ai__gemini-3.6-flash"
    );
  });

  it("includes hidden models when requested by an admin picker", () => {
    const hiddenModel = {
      ...makeModelConfiguration(11, "hidden-model"),
      is_visible: false,
    };
    const providers = [makeProvider(1, "OpenAI", "openai", [hiddenModel])];

    expect(buildLlmOptions(providers)).toHaveLength(0);
    expect(buildLlmOptions(providers, undefined, true)).toEqual([
      expect.objectContaining({ modelName: "hidden-model" }),
    ]);
  });
});
