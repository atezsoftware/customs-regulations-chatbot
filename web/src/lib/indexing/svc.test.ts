import {
  OPENROUTER_EMBEDDINGS_URL,
  connectOpenRouterEmbeddingProvider,
  deleteRerankingConfig,
  disconnectEmbeddingProvider,
  fetchOpenRouterEmbeddingModels,
  fetchOpenRouterRerankingModels,
  getRerankingConfig,
  saveRerankingConfig,
  setNewSearchSettings,
  testRerankingConfig,
} from "@/lib/indexing/svc";
import {
  EmbeddingProviderName,
  RerankingConfigView,
  SwitchoverType,
} from "@/lib/indexing/types";

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

describe("OpenRouter reranking administration service", () => {
  const fetchMock = jest.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    global.fetch = fetchMock;
  });

  it("reads the masked reranking configuration from the admin endpoint", async () => {
    const config: RerankingConfigView = {
      enabled: false,
      provider_type: "openrouter",
      model_id: "voyageai/rerank-2.5",
      api_key_configured: true,
      masked_api_key: "********",
    };
    fetchMock.mockResolvedValueOnce(jsonResponse(config));

    await expect(getRerankingConfig()).resolves.toEqual(config);
    expect(fetchMock).toHaveBeenCalledWith("/api/admin/reranking/config");

    type GetConfigHasPlaintextKey = "api_key" extends keyof RerankingConfigView
      ? true
      : false;
    const getConfigHasPlaintextKey: GetConfigHasPlaintextKey = false;
    expect(getConfigHasPlaintextKey).toBe(false);
  });

  it("omits an unchanged masked key when saving", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        enabled: false,
        provider_type: "openrouter",
        model_id: "voyageai/rerank-2.5",
        api_key_configured: true,
        masked_api_key: "********",
      })
    );

    await saveRerankingConfig({
      enabled: false,
      provider_type: "openrouter",
      model_id: "voyageai/rerank-2.5",
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/admin/reranking/config",
      expect.objectContaining({
        method: "PUT",
        body: JSON.stringify({
          enabled: false,
          provider_type: "openrouter",
          model_id: "voyageai/rerank-2.5",
        }),
      })
    );
  });

  it("sends a new key and exact test attestation when saving", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        enabled: true,
        provider_type: "openrouter",
        model_id: "voyageai/rerank-2.5",
        api_key_configured: true,
        masked_api_key: "********",
      })
    );

    await saveRerankingConfig({
      enabled: true,
      provider_type: "openrouter",
      model_id: "voyageai/rerank-2.5",
      api_key: "sk-or-unsaved",
      test_attestation: "attestation-token",
    });

    expect(requestBody(fetchMock, 0)).toEqual({
      enabled: true,
      provider_type: "openrouter",
      model_id: "voyageai/rerank-2.5",
      api_key: "sk-or-unsaved",
      test_attestation: "attestation-token",
    });
  });

  it("loads the reranker catalog with an unsaved key in the POST body", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        models: [
          {
            id: "voyageai/rerank-2.5",
            name: "Voyage Rerank 2.5",
          },
        ],
      })
    );

    await expect(
      fetchOpenRouterRerankingModels("sk-or-unsaved")
    ).resolves.toEqual([
      { id: "voyageai/rerank-2.5", name: "Voyage Rerank 2.5" },
    ]);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/admin/reranking/openrouter-models",
      expect.objectContaining({ method: "POST" })
    );
    expect(requestBody(fetchMock, 0)).toEqual({ api_key: "sk-or-unsaved" });
  });

  it("omits the key from a stored-key catalog lookup", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ models: [] }));

    await fetchOpenRouterRerankingModels();

    expect(requestBody(fetchMock, 0)).toEqual({});
  });

  it("tests an unsaved key and model and returns the attestation", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ success: true, test_attestation: "attestation-token" })
    );

    await expect(
      testRerankingConfig({
        provider_type: "openrouter",
        model_id: "voyageai/rerank-2.5",
        api_key: "sk-or-unsaved",
      })
    ).resolves.toEqual({
      success: true,
      test_attestation: "attestation-token",
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/admin/reranking/test",
      expect.objectContaining({ method: "POST" })
    );
    expect(requestBody(fetchMock, 0)).toEqual({
      provider_type: "openrouter",
      model_id: "voyageai/rerank-2.5",
      api_key: "sk-or-unsaved",
    });
  });

  it("omits unchanged key overrides when testing stored credentials", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ success: true, test_attestation: "attestation-token" })
    );

    await testRerankingConfig({
      provider_type: "openrouter",
      model_id: "voyageai/rerank-2.5",
    });

    expect(requestBody(fetchMock, 0)).toEqual({
      provider_type: "openrouter",
      model_id: "voyageai/rerank-2.5",
    });
  });

  it("deletes and purges the persisted reranker configuration", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(undefined));

    await expect(deleteRerankingConfig()).resolves.toBeUndefined();
    expect(fetchMock).toHaveBeenCalledWith("/api/admin/reranking/config", {
      method: "DELETE",
    });
  });

  it("surfaces backend detail for reranking failures", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(
        { detail: "Test this exact API key and model first." },
        false
      )
    );

    await expect(
      saveRerankingConfig({
        enabled: true,
        provider_type: "openrouter",
        model_id: "voyageai/rerank-2.5",
        test_attestation: "expired-token",
      })
    ).rejects.toThrow("Test this exact API key and model first.");
  });
});
