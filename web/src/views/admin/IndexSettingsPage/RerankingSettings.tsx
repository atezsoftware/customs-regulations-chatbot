"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Button,
  Card,
  InputTypeIn,
  MessageCard,
  PasswordInputTypeIn,
  Switch,
  Text,
  useCreateModal,
} from "@opal/components";
import {
  ConfirmationModalLayout,
  Content,
  InputHorizontal,
} from "@opal/layouts";
import { SvgTrash } from "@opal/icons";
import * as GeneralLayouts from "@/layouts/general-layouts";
import InputSelect from "@/refresh-components/inputs/InputSelect";
import { useRerankingConfig } from "@/lib/indexing/hooks";
import {
  deleteRerankingConfig,
  fetchOpenRouterRerankingModels,
  fetchSiliconFlowRerankingModels,
  saveRerankingConfig,
  testRerankingConfig,
} from "@/lib/indexing/svc";
import type {
  OpenRouterRerankingModel,
  RerankingProviderType,
} from "@/lib/indexing/types";

const EMPTY_KEY = "";
const DEFAULT_PROVIDER: RerankingProviderType = "siliconflow";
const DEFAULT_MODELS: Record<RerankingProviderType, string> = {
  openrouter: "voyageai/rerank-3",
  siliconflow: "Qwen/Qwen3-Reranker-8B",
};
const PROVIDER_LABELS: Record<RerankingProviderType, string> = {
  openrouter: "OpenRouter",
  siliconflow: "SiliconFlow",
};

function errorDetail(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "The reranking operation failed.";
}

export default function RerankingSettings() {
  const { data: persistedConfig, isLoading, mutate } = useRerankingConfig();
  const deleteModal = useCreateModal();
  const [enabled, setEnabled] = useState(persistedConfig?.enabled ?? false);
  const [provider, setProvider] = useState<RerankingProviderType>(
    persistedConfig?.provider_type ?? DEFAULT_PROVIDER
  );
  const [modelId, setModelId] = useState(
    persistedConfig?.model_id ??
      DEFAULT_MODELS[persistedConfig?.provider_type ?? DEFAULT_PROVIDER]
  );
  const [apiKey, setApiKey] = useState(EMPTY_KEY);
  const [catalog, setCatalog] = useState<OpenRouterRerankingModel[]>([]);
  const [attestation, setAttestation] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!persistedConfig) return;
    setEnabled(persistedConfig.enabled);
    setProvider(persistedConfig.provider_type ?? DEFAULT_PROVIDER);
    setModelId(
      persistedConfig.model_id ??
        DEFAULT_MODELS[persistedConfig.provider_type ?? DEFAULT_PROVIDER]
    );
    setApiKey(EMPTY_KEY);
    setCatalog([]);
    setAttestation(null);
  }, [persistedConfig]);

  const invalidateTest = useCallback(() => {
    setAttestation(null);
    setSuccessMessage(null);
  }, []);

  const handleApiKeyChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      setApiKey(event.target.value);
      invalidateTest();
    },
    [invalidateTest]
  );

  const handleModelChange = useCallback(
    (nextModelId: string) => {
      setModelId(nextModelId);
      invalidateTest();
    },
    [invalidateTest]
  );

  const handleProviderChange = useCallback(
    (nextProvider: string) => {
      if (nextProvider !== "openrouter" && nextProvider !== "siliconflow")
        return;
      setProvider(nextProvider);
      setModelId(
        persistedConfig?.provider_type === nextProvider
          ? (persistedConfig.model_id ?? DEFAULT_MODELS[nextProvider])
          : DEFAULT_MODELS[nextProvider]
      );
      setApiKey(EMPTY_KEY);
      setCatalog([]);
      setErrorMessage(null);
      invalidateTest();
    },
    [invalidateTest, persistedConfig]
  );

  const providerLabel = PROVIDER_LABELS[provider];
  const hasStoredKeyForProvider = Boolean(
    persistedConfig?.provider_type === provider &&
    persistedConfig.api_key_configured
  );
  const hasKey = apiKey.trim().length > 0 || hasStoredKeyForProvider;
  const needsNewProviderKey = Boolean(
    persistedConfig?.api_key_configured &&
    persistedConfig.provider_type !== provider &&
    !apiKey.trim()
  );

  const runOperation = useCallback(async (operation: () => Promise<void>) => {
    setIsBusy(true);
    setErrorMessage(null);
    try {
      await operation();
    } catch (error) {
      setSuccessMessage(null);
      setErrorMessage(errorDetail(error));
    } finally {
      setIsBusy(false);
    }
  }, []);

  const handleLoadModels = useCallback(() => {
    void runOperation(async () => {
      const models =
        provider === "openrouter"
          ? await fetchOpenRouterRerankingModels(apiKey.trim() || undefined)
          : await fetchSiliconFlowRerankingModels();
      setCatalog(models);
      setSuccessMessage(
        models.length > 0
          ? `${providerLabel} model catalog loaded`
          : `${providerLabel} returned no reranking models`
      );
    });
  }, [apiKey, provider, providerLabel, runOperation]);

  const handleTest = useCallback(() => {
    void runOperation(async () => {
      const response = await testRerankingConfig({
        provider_type: provider,
        model_id: modelId.trim(),
        ...(apiKey.trim() && { api_key: apiKey.trim() }),
      });
      if (!response.success) {
        throw new Error(`${providerLabel} did not confirm the reranking test.`);
      }
      setAttestation(response.test_attestation);
      setSuccessMessage("Configuration test passed");
    });
  }, [apiKey, modelId, provider, providerLabel, runOperation]);

  const handleSave = useCallback(() => {
    void runOperation(async () => {
      const saved = await saveRerankingConfig({
        enabled,
        provider_type: provider,
        model_id: modelId.trim(),
        ...(apiKey.trim() && { api_key: apiKey.trim() }),
        ...(enabled && attestation && { test_attestation: attestation }),
      });
      await mutate(saved, { revalidate: false });
      setSuccessMessage("Reranking configuration saved");
    });
  }, [apiKey, attestation, enabled, modelId, mutate, provider, runOperation]);

  const handleDelete = useCallback(() => {
    void runOperation(async () => {
      await deleteRerankingConfig();
      await mutate(
        {
          enabled: false,
          provider_type: null,
          model_id: null,
          api_key_configured: false,
          masked_api_key: null,
        },
        { revalidate: false }
      );
      setCatalog([]);
      deleteModal.toggle(false);
      setSuccessMessage("Reranking configuration deleted");
    });
  }, [deleteModal, mutate, runOperation]);

  const hasModel = modelId.trim().length > 0;
  const enabledSaveBlocked = enabled && attestation === null;
  const hasPersistedConfiguration = Boolean(
    persistedConfig?.provider_type ||
    persistedConfig?.model_id ||
    persistedConfig?.api_key_configured
  );

  return (
    <GeneralLayouts.Section
      gap={0.75}
      height="fit"
      alignItems="stretch"
      justifyContent="start"
    >
      <deleteModal.Provider>
        <ConfirmationModalLayout
          icon={SvgTrash}
          title="Delete reranking configuration"
          submit={
            <Button variant="danger" onClick={handleDelete} disabled={isBusy}>
              Delete and purge
            </Button>
          }
        >
          <Text font="main-ui-body" color="text-03" as="p">
            This permanently removes the encrypted reranker API key, model, and
            enabled state.
          </Text>
        </ConfirmationModalLayout>
      </deleteModal.Provider>

      <Content
        title="Reranking"
        description="Rerank the globally retrieved candidate pool before answer generation. This setting is saved independently and does not start a re-index."
        sizePreset="main-content"
        variant="section"
      />

      <MessageCard
        variant="warning"
        title="External data processing"
        description={`The query and authorized candidate text leave this deployment and are sent to ${providerLabel} for reranking.`}
        titleMaxLines={undefined}
      />

      <Card border="solid" rounding="lg">
        <GeneralLayouts.Section width="full" alignItems="stretch">
          <InputHorizontal
            title="Enable reranking"
            description="Enabled configurations affect standard search, Deep Search, ASv3, Legal Composite, and SuperSearch globally."
            withLabel
          >
            <Switch
              aria-label="Enable reranking"
              checked={enabled}
              disabled={isBusy || isLoading}
              onCheckedChange={setEnabled}
            />
          </InputHorizontal>

          <InputHorizontal
            title="Reranking provider"
            withLabel
            responsive
            fillInput
          >
            <InputSelect
              value={provider}
              onValueChange={handleProviderChange}
              disabled={isBusy || isLoading}
            >
              <InputSelect.Trigger aria-label="Reranking provider" />
              <InputSelect.Content>
                <InputSelect.Item value="openrouter">
                  OpenRouter
                </InputSelect.Item>
                <InputSelect.Item value="siliconflow">
                  SiliconFlow
                </InputSelect.Item>
              </InputSelect.Content>
            </InputSelect>
          </InputHorizontal>

          <InputHorizontal
            title={`${providerLabel} API key`}
            description={
              hasStoredKeyForProvider
                ? "A stored encrypted key is configured. Leave this blank to retain it."
                : `Enter your ${providerLabel} key. Keys from another provider are not reused.`
            }
            withLabel="reranking-api-key"
            responsive
            fillInput
          >
            <PasswordInputTypeIn
              id="reranking-api-key"
              aria-label={`${providerLabel} API key`}
              value={apiKey}
              placeholder={
                (hasStoredKeyForProvider && persistedConfig?.masked_api_key) ||
                `Enter your ${providerLabel} API key`
              }
              disabled={isBusy || isLoading}
              onChange={handleApiKeyChange}
            />
          </InputHorizontal>

          <GeneralLayouts.Section
            flexDirection="row"
            justifyContent="end"
            width="full"
          >
            <Button
              prominence="secondary"
              onClick={handleLoadModels}
              disabled={
                isBusy || isLoading || (provider === "openrouter" && !hasKey)
              }
            >
              Load models
            </Button>
          </GeneralLayouts.Section>

          {catalog.length > 0 && (
            <InputHorizontal
              title={`${providerLabel} model catalog`}
              description="Selecting a catalog entry copies its exact ID into the manual field below."
              withLabel
              responsive
              fillInput
            >
              <InputSelect
                value={
                  catalog.some((model) => model.id === modelId) ? modelId : ""
                }
                onValueChange={handleModelChange}
                disabled={isBusy || isLoading}
              >
                <InputSelect.Trigger
                  aria-label={`${providerLabel} reranking model catalog`}
                  placeholder="Select a discovered model"
                />
                <InputSelect.Content>
                  {catalog.map((model) => (
                    <InputSelect.Item key={model.id} value={model.id}>
                      {model.name}
                    </InputSelect.Item>
                  ))}
                </InputSelect.Content>
              </InputSelect>
            </InputHorizontal>
          )}

          <InputHorizontal
            title={`${providerLabel} model ID`}
            description={
              provider === "siliconflow"
                ? "Only SiliconFlow's supported Qwen3 reranker IDs are accepted."
                : "Use an exact OpenRouter reranking model ID from the catalog."
            }
            withLabel="reranking-model-id"
            responsive
            fillInput
          >
            <InputTypeIn
              id="reranking-model-id"
              aria-label={`${providerLabel} model ID`}
              value={modelId}
              placeholder={DEFAULT_MODELS[provider]}
              variant={isBusy || isLoading ? "disabled" : undefined}
              onChange={(event) => handleModelChange(event.target.value)}
            />
          </InputHorizontal>

          {errorMessage && (
            <MessageCard
              variant="error"
              title="Reranking operation failed"
              description={errorMessage}
              titleMaxLines={undefined}
            />
          )}
          {successMessage && (
            <MessageCard
              variant="success"
              title={successMessage}
              titleMaxLines={undefined}
            />
          )}

          <GeneralLayouts.Section
            flexDirection="row"
            justifyContent="end"
            width="full"
            gap={0.5}
          >
            <Button
              prominence="secondary"
              onClick={handleTest}
              disabled={isBusy || isLoading || !hasModel || !hasKey}
            >
              Test configuration
            </Button>
            <Button
              onClick={handleSave}
              disabled={
                isBusy ||
                isLoading ||
                !hasModel ||
                enabledSaveBlocked ||
                needsNewProviderKey
              }
            >
              {enabled ? "Enable and save" : "Save disabled configuration"}
            </Button>
            <Button
              variant="danger"
              prominence="secondary"
              onClick={() => deleteModal.toggle(true)}
              disabled={isBusy || isLoading || !hasPersistedConfiguration}
            >
              Delete configuration
            </Button>
          </GeneralLayouts.Section>
        </GeneralLayouts.Section>
      </Card>
    </GeneralLayouts.Section>
  );
}
