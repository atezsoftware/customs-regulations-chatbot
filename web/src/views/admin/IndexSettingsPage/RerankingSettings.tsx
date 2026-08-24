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
  fetchSiliconFlowRerankingModels,
  saveRerankingConfig,
  testRerankingConfig,
} from "@/lib/indexing/svc";
import type { OpenRouterRerankingModel } from "@/lib/indexing/types";

const EMPTY_KEY = "";
const DEFAULT_MODEL = "Qwen/Qwen3-Reranker-8B";

function errorDetail(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "The reranking operation failed.";
}

export default function RerankingSettings() {
  const { data: persistedConfig, isLoading, mutate } = useRerankingConfig();
  const deleteModal = useCreateModal();
  const [enabled, setEnabled] = useState(persistedConfig?.enabled ?? false);
  const [modelId, setModelId] = useState(
    persistedConfig?.provider_type === "siliconflow"
      ? (persistedConfig.model_id ?? DEFAULT_MODEL)
      : DEFAULT_MODEL
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
    setModelId(
      persistedConfig.provider_type === "siliconflow"
        ? (persistedConfig.model_id ?? DEFAULT_MODEL)
        : DEFAULT_MODEL
    );
    setApiKey(EMPTY_KEY);
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
      const models = await fetchSiliconFlowRerankingModels();
      setCatalog(models);
      setSuccessMessage(
        models.length > 0
          ? "SiliconFlow model catalog loaded"
          : "SiliconFlow returned no reranking models"
      );
    });
  }, [apiKey, runOperation]);

  const handleTest = useCallback(() => {
    void runOperation(async () => {
      const response = await testRerankingConfig({
        provider_type: "siliconflow",
        model_id: modelId.trim(),
        ...(apiKey.trim() && { api_key: apiKey.trim() }),
      });
      if (!response.success) {
        throw new Error("SiliconFlow did not confirm the reranking test.");
      }
      setAttestation(response.test_attestation);
      setSuccessMessage("Configuration test passed");
    });
  }, [apiKey, modelId, runOperation]);

  const handleSave = useCallback(() => {
    void runOperation(async () => {
      const saved = await saveRerankingConfig({
        enabled,
        provider_type: "siliconflow",
        model_id: modelId.trim(),
        ...(apiKey.trim() && { api_key: apiKey.trim() }),
        ...(enabled && attestation && { test_attestation: attestation }),
      });
      await mutate(saved, { revalidate: false });
      setSuccessMessage("Reranking configuration saved");
    });
  }, [apiKey, attestation, enabled, modelId, mutate, runOperation]);

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
            This permanently removes the encrypted SiliconFlow API key, model,
            and enabled state.
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
        description="The query and authorized candidate text leave this deployment and are sent to SiliconFlow for reranking."
        titleMaxLines={undefined}
      />

      <Card border="solid" rounding="lg">
        <GeneralLayouts.Section width="full" alignItems="stretch">
          <InputHorizontal
            title="Enable reranking"
            description="Enabled configurations affect standard search and Deep Search globally."
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
            title="SiliconFlow API key"
            description={
              persistedConfig?.api_key_configured
                ? "A stored encrypted key is configured. Leave this blank to retain it."
                : "Used only for SiliconFlow testing and reranking."
            }
            withLabel="reranking-api-key"
            responsive
            fillInput
          >
            <PasswordInputTypeIn
              id="reranking-api-key"
              aria-label="SiliconFlow API key"
              value={apiKey}
              placeholder={
                persistedConfig?.masked_api_key ?? "Enter a SiliconFlow API key"
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
              disabled={isBusy || isLoading}
            >
              Load models
            </Button>
          </GeneralLayouts.Section>

          {catalog.length > 0 && (
            <InputHorizontal
              title="SiliconFlow model catalog"
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
              >
                <InputSelect.Trigger
                  aria-label="SiliconFlow reranking model catalog"
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
            title="SiliconFlow model ID"
            description="Only SiliconFlow's supported Qwen3 reranker IDs are accepted."
            withLabel="reranking-model-id"
            responsive
            fillInput
          >
            <InputTypeIn
              id="reranking-model-id"
              aria-label="SiliconFlow model ID"
              value={modelId}
              placeholder={DEFAULT_MODEL}
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
              disabled={isBusy || isLoading || !hasModel}
            >
              Test configuration
            </Button>
            <Button
              onClick={handleSave}
              disabled={isBusy || isLoading || !hasModel || enabledSaveBlocked}
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
