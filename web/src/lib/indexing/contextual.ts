import type { ContextualSetupStatus } from "@/lib/indexing/types";

export function contextualSetupStatusAfterSave(
  currentStatus: ContextualSetupStatus | undefined,
  modelConfigurationId: number
): ContextualSetupStatus {
  return {
    required: currentStatus?.required ?? true,
    enabled: true,
    model_configuration_id: modelConfigurationId,
  };
}
