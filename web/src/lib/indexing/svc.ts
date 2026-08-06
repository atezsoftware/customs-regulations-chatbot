import type { Settings } from "@/lib/settings/types";
import { SWR_KEYS } from "@/lib/swr-keys";
import {
  EmbeddingModel,
  EmbeddingProviderName,
  OpenRouterRerankingModel,
  OpenRouterRerankingModelsResponse,
  OpenRouterEmbeddingModelResponse,
  ReindexErrorRow,
  RerankingConfigUpdate,
  RerankingConfigView,
  RerankingTestRequest,
  RerankingTestResponse,
  SavedSearchSettings,
  SwitchoverType,
} from "@/lib/indexing/types";
import { isCloudBased } from "@/lib/indexing";

export const OPENROUTER_EMBEDDINGS_URL =
  "https://openrouter.ai/api/v1/embeddings";

interface EmbeddingTestResponse {
  embedding_dimension: number;
}

interface SaveEmbeddingProviderCredentialsArgs {
  providerType: EmbeddingProviderName;
  apiKey: string | null;
  apiUrl: string;
  apiVersion: string | null;
  deploymentName: string | null;
}

interface TestEmbeddingArgs {
  provider_type: string;
  modelName: string;
  apiKey: string | null;
  apiUrl: string | null;
  apiVersion: string | null;
  deploymentName: string | null;
}

export async function testEmbedding({
  provider_type,
  modelName,
  apiKey,
  apiUrl,
  apiVersion,
  deploymentName,
}: TestEmbeddingArgs) {
  const testModelName =
    provider_type === "openai" ? "text-embedding-3-small" : modelName;

  return await fetch("/api/admin/embedding/test-embedding", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      provider_type: provider_type,
      api_key: apiKey,
      api_url: apiUrl,
      model_name: testModelName,
      api_version: apiVersion,
      deployment_name: deploymentName,
    }),
  });
}

async function responseErrorMessage(
  response: Response,
  fallback: string
): Promise<string> {
  try {
    const error = (await response.json()) as {
      detail?: string;
      message?: string;
    };
    return error.detail ?? error.message ?? fallback;
  } catch {
    return fallback;
  }
}

async function saveEmbeddingProviderCredentials({
  providerType,
  apiKey,
  apiUrl,
  apiVersion,
  deploymentName,
}: SaveEmbeddingProviderCredentialsArgs): Promise<void> {
  const body: Record<string, unknown> = {
    provider_type: providerType,
    api_url: apiUrl,
    api_version: apiVersion,
    deployment_name: deploymentName,
    is_default_provider: false,
    is_configured: true,
  };
  if (apiKey !== null) body.api_key = apiKey;

  const saveResponse = await fetch(SWR_KEYS.embeddingProviders, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!saveResponse.ok) {
    throw new Error(
      await responseErrorMessage(saveResponse, "Failed to save provider")
    );
  }
}

/** Fetches the embedding-only catalog exposed by the OpenRouter admin API. */
export async function fetchOpenRouterEmbeddingModels(
  apiKey: string | null
): Promise<OpenRouterEmbeddingModelResponse[]> {
  const response = await fetch(
    "/api/admin/embedding/openrouter/available-models",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: apiKey }),
    }
  );

  if (!response.ok) {
    throw new Error(
      await responseErrorMessage(response, "Failed to fetch OpenRouter models")
    );
  }

  return (await response.json()) as OpenRouterEmbeddingModelResponse[];
}

/** Reads the secret-free persisted OpenRouter reranker configuration. */
export async function getRerankingConfig(): Promise<RerankingConfigView> {
  const response = await fetch(SWR_KEYS.rerankingConfig);
  if (!response.ok) {
    throw new Error(
      await responseErrorMessage(
        response,
        "Failed to fetch reranking configuration"
      )
    );
  }
  return (await response.json()) as RerankingConfigView;
}

/**
 * Saves reranking configuration. An omitted key means retain the encrypted
 * credential already stored by the backend.
 */
export async function saveRerankingConfig(
  config: RerankingConfigUpdate
): Promise<RerankingConfigView> {
  const response = await fetch(SWR_KEYS.rerankingConfig, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      enabled: config.enabled,
      provider_type: config.provider_type,
      model_id: config.model_id,
      ...(config.api_key !== undefined && { api_key: config.api_key }),
      ...(config.test_attestation !== undefined && {
        test_attestation: config.test_attestation,
      }),
    }),
  });
  if (!response.ok) {
    throw new Error(
      await responseErrorMessage(
        response,
        "Failed to save reranking configuration"
      )
    );
  }
  return (await response.json()) as RerankingConfigView;
}

/** Disables reranking and purges its persisted credential. */
export async function deleteRerankingConfig(): Promise<void> {
  const response = await fetch(SWR_KEYS.rerankingConfig, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(
      await responseErrorMessage(
        response,
        "Failed to delete reranking configuration"
      )
    );
  }
}

/** Tests persisted or unsaved OpenRouter credentials without storing overrides. */
export async function testRerankingConfig(
  request: RerankingTestRequest
): Promise<RerankingTestResponse> {
  const response = await fetch(SWR_KEYS.rerankingTest, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      provider_type: request.provider_type,
      ...(request.model_id !== undefined && { model_id: request.model_id }),
      ...(request.api_key !== undefined && { api_key: request.api_key }),
    }),
  });
  if (!response.ok) {
    throw new Error(
      await responseErrorMessage(
        response,
        "Reranking configuration test failed"
      )
    );
  }
  return (await response.json()) as RerankingTestResponse;
}

/** Loads OpenRouter's reranking catalog using an optional unsaved key. */
export async function fetchOpenRouterRerankingModels(
  apiKey?: string
): Promise<OpenRouterRerankingModel[]> {
  const response = await fetch(SWR_KEYS.openRouterRerankingModels, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...(apiKey !== undefined && { api_key: apiKey }),
    }),
  });
  if (!response.ok) {
    throw new Error(
      await responseErrorMessage(
        response,
        "Failed to fetch OpenRouter reranking models"
      )
    );
  }
  const result = (await response.json()) as OpenRouterRerankingModelsResponse;
  return result.models;
}

/**
 * Tests and persists the fixed OpenRouter embedding connection.
 * The test response is authoritative for model dimension, so the UI never
 * asks an administrator to enter that value manually.
 */
export async function connectOpenRouterEmbeddingProvider({
  apiKey,
  modelName,
}: {
  apiKey: string | null;
  modelName: string;
}): Promise<number> {
  const testResponse = await testEmbedding({
    provider_type: EmbeddingProviderName.OPENROUTER,
    modelName,
    apiKey,
    apiUrl: OPENROUTER_EMBEDDINGS_URL,
    apiVersion: null,
    deploymentName: null,
  });

  if (!testResponse.ok) {
    throw new Error(
      await responseErrorMessage(testResponse, "Embedding test failed")
    );
  }

  const result = (await testResponse.json()) as Partial<EmbeddingTestResponse>;
  if (
    !Number.isInteger(result.embedding_dimension) ||
    (result.embedding_dimension ?? 0) <= 0
  ) {
    throw new Error("OpenRouter returned an invalid embedding dimension");
  }

  await saveEmbeddingProviderCredentials({
    providerType: EmbeddingProviderName.OPENROUTER,
    apiKey,
    apiUrl: OPENROUTER_EMBEDDINGS_URL,
    apiVersion: null,
    deploymentName: null,
  });

  return result.embedding_dimension as number;
}

/**
 * Tests and saves embedding provider credentials.
 * Tests the connection first, then persists the credentials.
 * Throws on failure with a user-facing error message.
 *
 * `apiVersion` and `deploymentName` are Azure-specific — backend's
 * `CloudEmbeddingProviderCreationRequest` accepts them as optional, and
 * non-Azure providers should pass `null`.
 */
export async function connectEmbeddingProvider({
  providerType,
  apiKey,
  apiUrl,
  modelName = "",
  apiVersion,
  deploymentName,
}: {
  providerType: EmbeddingProviderName;
  apiKey: string | null;
  apiUrl: string;
  modelName?: string;
  apiVersion: string | null;
  deploymentName: string | null;
}): Promise<void> {
  if (apiKey !== null) {
    const testResponse = await testEmbedding({
      provider_type: providerType,
      modelName,
      apiKey,
      apiUrl,
      apiVersion,
      deploymentName,
    });

    if (!testResponse.ok) {
      const err = await testResponse.json();
      throw new Error(err.detail ?? "Embedding test failed");
    }
  }

  await saveEmbeddingProviderCredentials({
    providerType,
    apiKey,
    apiUrl,
    apiVersion,
    deploymentName,
  });
}

/**
 * Disconnects an embedding provider by deleting its credentials.
 * Throws on failure with a user-facing error message.
 */
export async function disconnectEmbeddingProvider(
  providerType: string
): Promise<void> {
  const response = await fetch(
    `${SWR_KEYS.embeddingProviders}/${providerType}`,
    { method: "DELETE" }
  );

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail ?? "Failed to disconnect provider");
  }
}

export async function saveAdminSettings(settings: Settings) {
  const response = await fetch("/api/admin/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  });

  if (!response.ok) {
    const errorMsg = (await response.json()).detail;
    throw new Error(errorMsg);
  }
}

/**
 * Cancels an in-flight embedding-model switchover. Marks the FUTURE search
 * settings row as PAST, expires its index attempts, and drops the secondary
 * vector index.
 */
export async function cancelNewEmbedding(): Promise<Response> {
  return await fetch("/api/search-settings/cancel-new-embedding", {
    method: "POST",
  });
}

/**
 * Resume a paused re-index unit from its cursor. Throws on a hard failure; a 503
 * (resumed but the queue is down) is treated as success — the scheduler re-dispatches it.
 */
export async function resumePausedPort(
  row: Pick<ReindexErrorRow, "cc_pair_id" | "user_id">
): Promise<void> {
  const response = await fetch("/api/search-settings/reindex/port/resume", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cc_pair_id: row.cc_pair_id, user_id: row.user_id }),
  });
  if (response.status === 503) {
    return;
  }
  if (!response.ok) {
    let detail: string | undefined;
    try {
      detail = ((await response.json()) as { detail?: string }).detail;
    } catch (e) {
      // non-JSON error body (e.g. a 502 HTML page): log so the failure is traceable
      console.error(`resumePausedPort failed (${response.status}):`, e);
    }
    throw new Error(detail ?? "Failed to resume the paused unit.");
  }
}

interface SetNewSearchSettingsArgs {
  model: EmbeddingModel;
  providerName: EmbeddingProviderName;
  switchoverType: SwitchoverType;
  enableContextualRag: boolean;
  contextualRagModelConfigurationId: number | null;
}

export async function setNewSearchSettings({
  model,
  providerName,
  switchoverType,
  enableContextualRag,
  contextualRagModelConfigurationId,
}: SetNewSearchSettingsArgs): Promise<Response> {
  // The backend's EmbeddingProvider enum only contains cloud providers
  // (openai/cohere/voyage/google/openrouter/litellm/azure). Self-hosted models live
  // under the frontend's EmbeddingProviderName for UI grouping (icon,
  // docs link), but the backend expects provider_type=null for them.
  const providerType = isCloudBased(providerName) ? providerName : null;

  return await fetch("/api/search-settings/set-new-search-settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_name: model.modelName,
      model_dim: model.modelDim,
      normalize: model.normalize,
      query_prefix: model.queryPrefix,
      passage_prefix: model.passagePrefix,
      provider_type: providerType,
      api_key: null,
      api_url: null,
      index_name: null,
      multipass_indexing: false,
      enable_contextual_rag: enableContextualRag,
      contextual_rag_model_configuration_id: contextualRagModelConfigurationId,
      switchover_type: switchoverType,
    }),
  });
}

/**
 * Persists non-reindex search-settings updates (e.g. toggling Contextual RAG
 * or switching its LLM). Backend is `update_saved_search_settings` — it
 * mutates the CURRENT search-settings row in place rather than creating a new
 * one + kicking off a re-index. Caller is responsible for ensuring the
 * embedding-model fields in `settings` match the current model; the endpoint
 * does not validate this.
 */
export async function updateInferenceSettings(
  settings: SavedSearchSettings
): Promise<Response> {
  return await fetch("/api/search-settings/update-inference-settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  });
}
