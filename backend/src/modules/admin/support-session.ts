import type {AdminSessionRow} from '../chat/repositories/chat-session.repository';

export interface AdminSupportModelSelection {
  provider: string;
  modelId: string;
}

export function toAdminSupportModel(
  provider: string | null | undefined,
  modelId: string | null | undefined,
): AdminSupportModelSelection | undefined {
  if (!modelId) return undefined;
  return {provider: provider || 'unknown', modelId};
}

export function toAdminSupportSession(
  row: Pick<AdminSessionRow, 'llm_provider' | 'model'>,
): {model?: AdminSupportModelSelection} {
  const model = toAdminSupportModel(row.llm_provider, row.model);
  return model ? {model} : {};
}
