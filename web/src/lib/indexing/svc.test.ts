import {
  OPENROUTER_EMBEDDINGS_URL,
  connectOpenRouterEmbeddingProvider,
  disconnectEmbeddingProvider,
  fetchOpenRouterEmbeddingModels,
  setNewSearchSettings,
} from "@/lib/indexing/svc";
import { EmbeddingProviderName, SwitchoverType } from "@/lib/indexing/types";

function jsonResponse(body: unknown, ok = true): Response {
  return {
    ok,
    json: async () => body,
  } as Response;
}

function requestBody(fetchMock: jest.Mock, callIndex: number): unknown {
  const init = fetchMock.mock.calls[callIndex]?.[1] as RequestInit | undefined;
  return JSON.parse(String(init?.body));
}

describe("OpenRouter embedding service", () => {
  const fetchMock = jest.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    global.fetch = fetchMock;
  });

  it("fetches the embedding catalog with only the API key", async () => {
    const models = [
      {
        name: "openai/text-embedding-3-small",
        display_name: "Text Embedding 3 Small",
        description: "Fast embedding model",
        context_length: 8191,
      },
    ];
    fetchMock.mockResolvedValueOnce(jsonResponse(models));

    await expect(fetchOpenRouterEmbeddingModels("sk-or-test")).resolves.toEqual(
      models
    );
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/admin/embedding/openrouter/available-models",
      expect.objectContaining({ method: "POST" })
    );
    expect(requestBody(fetchMock, 0)).toEqual({ api_key: "sk-or-test" });
  });

  it("tests the selected model, uses its returned dimension, and saves the fixed URL", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ embedding_dimension: 1536 }))
      .mockResolvedValueOnce(jsonResponse({}));

    await expect(
      connectOpenRouterEmbeddingProvider({
        apiKey: "sk-or-test",
        modelName: "openai/text-embedding-3-small",
      })
    ).resolves.toBe(1536);

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/api/admin/embedding/test-embedding",
      expect.objectContaining({ method: "POST" })
    );
    expect(requestBody(fetchMock, 0)).toEqual({
      provider_type: "openrouter",
      api_key: "sk-or-test",
      api_url: OPENROUTER_EMBEDDINGS_URL,
      model_name: "openai/text-embedding-3-small",
      api_version: null,
      deployment_name: null,
    });
    expect(requestBody(fetchMock, 1)).toEqual({
      provider_type: "openrouter",
      api_url: OPENROUTER_EMBEDDINGS_URL,
      api_version: null,
      deployment_name: null,
      is_default_provider: false,
      is_configured: true,
      api_key: "sk-or-test",
    });
  });

  it("does not save credentials when the test omits a valid dimension", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({}));

    await expect(
      connectOpenRouterEmbeddingProvider({
        apiKey: "sk-or-test",
        modelName: "invalid-model",
      })
    ).rejects.toThrow("invalid embedding dimension");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("uses the stored API key when editing without resending the masked value", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ embedding_dimension: 768 }))
      .mockResolvedValueOnce(jsonResponse({}));

    await connectOpenRouterEmbeddingProvider({
      apiKey: null,
      modelName: "google/gemini-embedding-001",
    });

    expect(requestBody(fetchMock, 0)).toEqual(
      expect.objectContaining({
        provider_type: "openrouter",
        api_key: null,
      })
    );
    expect(requestBody(fetchMock, 1)).not.toHaveProperty("api_key");
  });

  it("disconnects the dedicated OpenRouter provider directly", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({}));

    await disconnectEmbeddingProvider(EmbeddingProviderName.OPENROUTER);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/admin/embedding/embedding-provider/openrouter",
      { method: "DELETE" }
    );
  });

  it("sends OpenRouter directly when applying new search settings", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({}));

    await setNewSearchSettings({
      model: {
        modelName: "openai/text-embedding-3-small",
        modelDim: 1536,
        normalize: false,
        queryPrefix: null,
        passagePrefix: null,
        description: "",
      },
      providerName: EmbeddingProviderName.OPENROUTER,
      switchoverType: SwitchoverType.REINDEX,
      enableContextualRag: false,
      contextualRagModelConfigurationId: null,
    });

    expect(requestBody(fetchMock, 0)).toEqual(
      expect.objectContaining({
        provider_type: "openrouter",
        model_name: "openai/text-embedding-3-small",
        model_dim: 1536,
      })
    );
  });
});
