import type {LlmModelOption} from '../../types';

export function applyModelSelection(
  model: LlmModelOption,
  onChange: (model: LlmModelOption) => void,
): {isOpen: false; query: ''} {
  onChange(model);
  return {isOpen: false, query: ''};
}
