"use client";

import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import {
  MAX_MODELS,
  SelectedModel,
} from "@/sections/model-selector/MultiModelSelector";
import { LLMOverride } from "@/app/app/services/lib";
import { LlmManager } from "@/lib/hooks";
import { buildLlmOptions, llmOptionKey } from "@/lib/languageModels/options";
import { getSelectableLlmProviders } from "@/lib/languageModels/utils";

export interface UseMultiModelChatReturn {
  /** Currently selected models for multi-model comparison. */
  selectedModels: SelectedModel[];
  /** Whether multi-model mode is active (>1 model selected). */
  isMultiModelActive: boolean;
  /** Add a model to the selection. */
  addModel: (model: SelectedModel) => void;
  /** Remove a model by index. */
  removeModel: (index: number) => void;
  /** Replace a model at a specific index with a new one. */
  replaceModel: (index: number, model: SelectedModel) => void;
  /** Clear all selected models. */
  clearModels: () => void;
  /** Build the LLMOverride[] array from selectedModels. */
  buildLlmOverrides: () => LLMOverride[];
  /**
   * Restore multi-model selection from model version strings (e.g. from chat history).
   * Matches against available llmOptions to reconstruct full SelectedModel objects.
   * History stores only name strings, so when two providers expose a model with the
   * same name this resolves to the first match.
   */
  restoreFromModelNames: (modelNames: string[]) => void;
  /**
   * Switch to a single model by name (after user picks a preferred response).
   * Matches against llmOptions to find the full SelectedModel. Name-only, so
   * same-named models across providers resolve to the first match.
   */
  selectSingleModel: (modelName: string) => void;
}

export default function useMultiModelChat(
  llmManager: LlmManager,
  activeChatSessionId: string | null = null
): UseMultiModelChatReturn {
  const [selectedModels, setSelectedModels] = useState<SelectedModel[]>([]);
  const prevActiveSessionIdRef = useRef(activeChatSessionId);
  const startingNewChat =
    prevActiveSessionIdRef.current !== null && activeChatSessionId === null;
  useEffect(() => {
    prevActiveSessionIdRef.current = activeChatSessionId;
  }, [activeChatSessionId]);

  // Eligibility is tied to a provider/model identity; sharing a raw model
  // name with the current selection does not make another hidden row valid.
  const llmOptions = useMemo(
    () =>
      llmManager.llmProviders
        ? buildLlmOptions(
            getSelectableLlmProviders(
              llmManager.llmProviders,
              llmManager.defaultText
            ),
            undefined,
            true
          )
        : [],
    [llmManager.llmProviders, llmManager.defaultText]
  );

  // In single-model mode, derive the displayed model directly from
  // llmManager.currentLlm so it always stays in sync (no stale state).
  // Only use the selectedModels state array when the user has manually
  // added multiple models (multi-model mode).
  const currentLlmModel = useMemo((): SelectedModel | null => {
    if (llmOptions.length === 0) return null;
    const { currentLlm } = llmManager;
    if (!currentLlm.modelName) return null;
    // Two providers can expose a model with the same name; prefer the one
    // whose provider (instance) name matches the current descriptor.
    const candidates = llmOptions.filter(
      (opt) =>
        opt.provider === currentLlm.provider &&
        opt.modelName === currentLlm.modelName
    );
    const match =
      candidates.find((opt) => opt.name === currentLlm.name) ?? candidates[0];
    if (!match) return null;
    return {
      name: match.name,
      provider: match.provider,
      modelName: match.modelName,
      modelConfigurationId: match.modelConfigurationId ?? null,
      displayName: match.displayName,
    };
  }, [llmOptions, llmManager.currentLlm]);

  const reconcileModels = useCallback(
    (models: SelectedModel[]): SelectedModel[] => {
      if (llmManager.llmProviders === undefined) return models;
      return models.flatMap((model) => {
        const current = llmOptions.find((option) =>
          model.modelConfigurationId != null
            ? option.modelConfigurationId === model.modelConfigurationId
            : option.name === model.name &&
              option.provider === model.provider &&
              option.modelName === model.modelName
        );
        return current
          ? [
              {
                name: current.name,
                provider: current.provider,
                modelName: current.modelName,
                modelConfigurationId: current.modelConfigurationId ?? null,
                displayName: current.displayName,
              },
            ]
          : [];
      });
    },
    [llmOptions, llmManager.llmProviders]
  );
  const reconciledModels = useMemo(
    () => (startingNewChat ? [] : reconcileModels(selectedModels)),
    [startingNewChat, reconcileModels, selectedModels]
  );

  // Persist removals so a model that reappears is not silently reselected.
  // If only one comparison model survives, make it the single-chat model too.
  useEffect(() => {
    if (reconciledModels.length === selectedModels.length) return;
    if (selectedModels.length > 1 && reconciledModels.length === 1) {
      const survivor = reconciledModels[0]!;
      llmManager.updateCurrentLlm({
        name: survivor.name,
        provider: survivor.provider,
        modelName: survivor.modelName,
      });
    }
    setSelectedModels(reconciledModels);
  }, [reconciledModels, selectedModels, llmManager]);

  const effectiveSelectedModels = useMemo(
    () =>
      selectedModels.length > 1 && reconciledModels.length > 0
        ? reconciledModels
        : currentLlmModel
          ? [currentLlmModel]
          : [],
    [selectedModels.length, reconciledModels, currentLlmModel]
  );
  const isMultiModelActive = effectiveSelectedModels.length > 1;

  const addModel = useCallback(
    (model: SelectedModel) => {
      setSelectedModels((prev) => {
        const available = reconcileModels(prev);
        const base =
          prev.length > 1 && available.length > 0
            ? available
            : currentLlmModel
              ? [currentLlmModel]
              : [];
        if (base.length >= MAX_MODELS) return base;
        if (base.some((m) => llmOptionKey(m) === llmOptionKey(model))) {
          return base;
        }
        return [...base, model];
      });
    },
    [currentLlmModel, reconcileModels]
  );

  const removeModel = useCallback(
    (index: number) => {
      const next = effectiveSelectedModels.filter((_, i) => i !== index);
      // When dropping to single-model, switch llmManager to the surviving
      // model so it becomes the active model instead of reverting to the
      // user's default.
      if (next.length === 1 && next[0]) {
        llmManager.updateCurrentLlm({
          name: next[0].name,
          provider: next[0].provider,
          modelName: next[0].modelName,
        });
      }
      setSelectedModels(next);
    },
    [effectiveSelectedModels, llmManager]
  );

  const replaceModel = useCallback(
    (index: number, model: SelectedModel) => {
      // In single-model mode, update llmManager directly so currentLlm
      // (and thus effectiveSelectedModels) reflects the change immediately.
      if (!isMultiModelActive) {
        llmManager.updateCurrentLlm({
          name: model.name,
          provider: model.provider,
          modelName: model.modelName,
        });
        return;
      }
      const target = effectiveSelectedModels[index];
      if (!target) return;
      setSelectedModels((prev) => {
        const available = reconcileModels(prev);
        const targetIndex = available.findIndex(
          (item) => llmOptionKey(item) === llmOptionKey(target)
        );
        if (targetIndex < 0) return available;
        // Don't replace with a model that's already selected elsewhere
        if (
          available.some(
            (m, i) =>
              i !== targetIndex && llmOptionKey(m) === llmOptionKey(model)
          )
        ) {
          return available;
        }
        const next = [...available];
        next[targetIndex] = model;
        return next;
      });
    },
    [isMultiModelActive, llmManager, effectiveSelectedModels, reconcileModels]
  );

  const clearModels = useCallback(() => {
    setSelectedModels([]);
  }, []);

  const restoreFromModelNames = useCallback(
    (modelNames: string[]) => {
      if (modelNames.length < 2 || llmOptions.length === 0) return;
      const restored: SelectedModel[] = [];
      for (const name of modelNames) {
        // Try matching by modelName (raw version string like "claude-opus-4-6")
        // or by displayName (friendly name like "Claude Opus 4.6")
        const match = llmOptions.find(
          (opt) =>
            opt.modelName === name ||
            opt.displayName === name ||
            opt.name === name
        );
        if (match) {
          restored.push({
            name: match.name,
            provider: match.provider,
            modelName: match.modelName,
            modelConfigurationId: match.modelConfigurationId ?? null,
            displayName: match.displayName,
          });
        }
      }
      if (restored.length >= 2) {
        setSelectedModels(restored.slice(0, MAX_MODELS));
      }
    },
    [llmOptions]
  );

  const selectSingleModel = useCallback(
    (modelName: string) => {
      if (llmOptions.length === 0) return;
      const match = llmOptions.find(
        (opt) =>
          opt.modelName === modelName ||
          opt.displayName === modelName ||
          opt.name === modelName
      );
      if (match) {
        setSelectedModels([
          {
            name: match.name,
            provider: match.provider,
            modelName: match.modelName,
            modelConfigurationId: match.modelConfigurationId ?? null,
            displayName: match.displayName,
          },
        ]);
      }
    },
    [llmOptions]
  );

  const buildLlmOverrides = useCallback((): LLMOverride[] => {
    return effectiveSelectedModels.map((m) => ({
      model_provider: m.name,
      model_provider_type: m.provider,
      model_version: m.modelName,
      display_name: m.displayName,
    }));
  }, [effectiveSelectedModels]);

  return {
    selectedModels: effectiveSelectedModels,
    isMultiModelActive,
    addModel,
    removeModel,
    replaceModel,
    clearModels,
    buildLlmOverrides,
    restoreFromModelNames,
    selectSingleModel,
  };
}
