import type { ScopedMutator, SWRConfiguration } from "swr";
import { SWR_KEYS } from "@/lib/swr-keys";

const PERSONA_PROVIDER_ENDPOINT_PATTERN =
  /^\/api\/llm\/persona\/\d+\/providers$/;

/** Keep open model selectors current while sharing requests across consumers. */
export const LLM_PROVIDER_REFRESH_OPTIONS = {
  refreshInterval: 60_000,
  revalidateOnFocus: true,
  revalidateOnReconnect: true,
  revalidateIfStale: true,
  focusThrottleInterval: 60_000,
  dedupingInterval: 5_000,
} satisfies SWRConfiguration;

export async function refreshLlmProviderCaches(
  mutate: ScopedMutator
): Promise<void> {
  await Promise.all([
    mutate(SWR_KEYS.adminLlmProviders),
    mutate(SWR_KEYS.llmProviders),
    mutate(SWR_KEYS.llmProvidersWithImageGen),
    mutate(
      (key) =>
        typeof key === "string" && PERSONA_PROVIDER_ENDPOINT_PATTERN.test(key)
    ),
  ]);
}
