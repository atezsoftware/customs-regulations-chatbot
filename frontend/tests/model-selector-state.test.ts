import assert from 'node:assert/strict';
import test from 'node:test';
import {applyModelSelection} from '../src/components/chat/model-selector-state';

test('selecting a model closes the selector and clears its search', () => {
  const model = {
    provider: 'openrouter',
    modelId: 'google/gemini-3-flash-preview',
    displayName: 'Gemini Flash',
    contextLength: 1,
    inputModalities: ['text'],
    outputModalities: ['text'],
    supportsReasoning: false,
    promptUsdPerMillion: '0.1',
    completionUsdPerMillion: '0.2',
    requestUsd: null,
  };
  let selectedModelId: string | undefined;

  const nextState = applyModelSelection(model, selected => {
    selectedModelId = selected.modelId;
  });

  assert.equal(selectedModelId, 'google/gemini-3-flash-preview');
  assert.deepEqual(nextState, {isOpen: false, query: ''});
});
