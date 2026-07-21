import {inject} from '@loopback/core';
import {repository} from '@loopback/repository';
import {get, HttpErrors, response} from '@loopback/rest';
import {LlmModelRepository} from '../repositories/llm-model.repository';

const DEFAULT_MODEL = 'google/gemini-3-flash-preview';

export class LlmModelsController {
  constructor(@repository(LlmModelRepository) private models: LlmModelRepository) {}

  @get('/llm/models')
  @response(200, {description: 'Compatible OpenRouter models safe for the model picker'})
  async list() {
    const [models, lastSyncedAt] = await Promise.all([this.models.activeModels(), this.models.lastSuccessfulSync()]);
    if (!models.length || !lastSyncedAt) throw new HttpErrors.ServiceUnavailable('Model catalog is temporarily unavailable.');
    return {
      defaultModelId: DEFAULT_MODEL,
      lastSyncedAt,
      models: models.map(model => ({
        provider: model.provider,
        modelId: model.modelId,
        displayName: model.displayName,
        description: model.description,
        contextLength: model.contextLength,
        maxCompletionTokens: model.maxCompletionTokens,
        inputModalities: model.inputModalities ?? [],
        outputModalities: model.outputModalities ?? [],
        supportsReasoning: Boolean((model.rawPricing as Record<string, unknown> | undefined)?.reasoning),
        promptUsdPerMillion: model.promptUsdPerToken ? String(Number(model.promptUsdPerToken) * 1_000_000) : null,
        completionUsdPerMillion: model.completionUsdPerToken ? String(Number(model.completionUsdPerToken) * 1_000_000) : null,
        requestUsd: model.requestUsd ?? null,
      })),
    };
  }
}
