import { fetchModels } from "@/lib/languageModels/svc";
import { LLMProviderName } from "@/lib/languageModels/types";

describe("Vertex model discovery", () => {
  afterEach(() => jest.restoreAllMocks());

  it("fetches the saved connection and exposes models absent from the static catalog", async () => {
    const fetchSpy = jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      json: async () => [
        {
          name: "gemini-future-flash",
          display_name: "Future Flash",
          max_input_tokens: 1000000,
          supports_image_input: true,
          supports_reasoning: true,
        },
      ],
    } as Response);
    const signal = new AbortController().signal;
    const result = await fetchModels(
      LLMProviderName.VERTEX_AI,
      {
        id: 7,
        custom_config: {
          vertex_credentials: "****",
          vertex_location: "global",
        },
      },
      signal
    );
    expect(result.error).toBeUndefined();
    expect(result.models).toEqual([
      {
        name: "gemini-future-flash",
        display_name: "Future Flash",
        effectiveDisplayName: "Future Flash",
        max_input_tokens: 1000000,
        supports_image_input: true,
        supports_reasoning: true,
        is_visible: false,
      },
    ]);
    expect(fetchSpy).toHaveBeenCalledWith(
      "/api/admin/llm/vertex-ai/available-models",
      expect.objectContaining({
        method: "POST",
        signal,
        body: JSON.stringify({
          provider_id: 7,
          custom_config: {
            vertex_credentials: "****",
            vertex_location: "global",
          },
        }),
      })
    );
  });

  it("returns discovery errors instead of falling back to static models", async () => {
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: false,
      json: async () => ({ detail: "Google permissions are missing" }),
    } as Response);
    const result = await fetchModels(LLMProviderName.VERTEX_AI, { id: 7 });
    expect(result).toEqual({
      models: [],
      error: "Google permissions are missing",
    });
  });
});
