"use client";

import useSWR from "swr";
import { Text } from "@opal/components";
import { errorHandlingFetcher } from "@/lib/fetcher";
import type { ResponseUsage } from "@/sections/chat/usage/interfaces";

const categories: Record<string, string> = {
  input: "Input",
  output: "Output",
  reasoning: "Reasoning",
  cache_read: "Cache okuma",
  cache_write: "Cache yazma",
  cache_write_1h: "Cache yazma (1 saat)",
  cache_write_unknown: "Cache yazma",
};
const sources: Record<string, string> = {
  model_registry: "Model fiyat kataloğu",
  provider_catalog: "Sağlayıcı fiyat kataloğu",
  admin_override: "Tanımlı kurum fiyatı",
  unavailable: "Fiyat kaydı yok",
};
// Mirrors the workflow button labels in AppInputBar so an answer reads with the
// same name the user picked. Keys come from backend onyx.chat.response_workflow.
const workflowLabels: Record<string, string> = {
  asv3: "ASv3 Tek Araştırma",
  asv3_deep: "ASv3 Deep",
  experimental_asv3: "Experimental ASv3",
  experimental: "Experimental",
  asv3_exp_cropped: "exp cropped",
  experimental_guardrails: "Experimental Guardrails",
  experimental_guardrails_v2: "Experimental Guardrails v2",
  experimental_guardrails_v3: "Experimental Guardrails v3",
  legal_review: "Hukuki İnceleme (Flash)",
  legal_composite: "Legal Composite",
  supersearch: "Supersearch",
  deep_research: "Deep Research",
  asv2: "Atez Search v2",
  asv1: "Atez Search",
};
export function workflowLabel(
  workflow: string | null | undefined
): string | null {
  if (!workflow) return null;
  return workflowLabels[workflow] ?? workflow;
}
export function formatResponseDuration(
  seconds: number | null | undefined
): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0)
    return "Süre kaydı yok";
  const whole = Math.round(seconds);
  return `${Math.floor(whole / 60)} dk ${whole % 60} sn`;
}
export function formatCost(usd: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 4,
    maximumFractionDigits: 6,
  }).format(usd);
}
export function costLabel(usage: ResponseUsage | null | undefined): string {
  if (!usage) return "Maliyet kaydı yok";
  if (usage.total_cost_usd != null)
    return `Tahmini LLM maliyeti ${formatCost(usage.total_cost_usd)}`;
  if (usage.known_cost_usd != null)
    return `En az ${formatCost(usage.known_cost_usd)} · maliyet eksik`;
  return "Maliyet hesaplanamadı";
}
interface ResponseUsageViewProps {
  usage: ResponseUsage | null | undefined;
  fallbackDuration?: number;
  compact?: boolean;
}
export function ResponseUsageView({
  usage,
  fallbackDuration,
  compact = false,
}: ResponseUsageViewProps) {
  const workflow = workflowLabel(usage?.workflow);
  const label = [
    workflow,
    formatResponseDuration(usage?.duration_seconds ?? fallbackDuration),
    costLabel(usage),
  ]
    .filter(Boolean)
    .join(" · ");
  if (compact || !usage?.models.length)
    return (
      <Text font="secondary-body" color="text-03">
        {label}
      </Text>
    );
  return (
    <details
      className="rounded-12 border border-border-02 px-3 py-2"
      data-testid="response-usage"
    >
      <summary className="cursor-pointer">
        <Text font="secondary-body" color="text-03">
          {label}
        </Text>
      </summary>
      <div className="flex flex-col gap-3 pt-2">
        <Text font="secondary-body" color="text-03">
          {`${usage.calls} LLM çağrısı; araştırma ve denetim çağrıları dahildir. Token × birim fiyat / 1.000.000. Reasoning, output toplamından ayrılarak bir kez hesaplanır. Sağlayıcı faturası, vergi ve ek hizmet ücretleri bu tahmine dahil değildir.`}
        </Text>
        {!!usage.excluded_service_calls && (
          <Text font="secondary-body" color="text-03">
            {`${usage.excluded_service_calls} embedding, reranker veya diğer model hizmeti çağrısı LLM token toplamına dahil değildir.`}
          </Text>
        )}
        {usage.unpriced_calls > 0 && (
          <Text font="secondary-body" color="text-03">
            {`${usage.unpriced_calls} çağrının fiyat veya token kaydı eksik.`}
          </Text>
        )}
        {usage.models.map((model, index) => (
          <div
            key={`${model.model}-${model.provider}-${index}`}
            className="flex flex-col gap-1"
          >
            <Text font="secondary-body">
              {`${model.model} · ${model.provider || "Sağlayıcı kaydı yok"}`}
            </Text>
            <Text font="secondary-body" color="text-03">
              {`${sources[model.source] || model.source} · ${new Date(model.priced_at).toLocaleString("tr-TR")}`}
            </Text>
            {model.lines.map((line) => (
              <Text key={line.category} font="secondary-body" color="text-03">
                {`${categories[line.category] || line.category}: ${line.tokens.toLocaleString("tr-TR")} token × ${
                  line.usd_per_million == null
                    ? "fiyat bilinmiyor"
                    : `${formatCost(line.usd_per_million)} / 1M`
                } = ${
                  line.cost_usd == null
                    ? "hesaplanamadı"
                    : formatCost(line.cost_usd)
                }`}
              </Text>
            ))}
            {!model.lines.length && (
              <Text font="secondary-body" color="text-03">
                {`Input: ${model.input_tokens ?? "bilinmiyor"} · Output: ${model.output_tokens ?? "bilinmiyor"} · Reasoning: ${model.reasoning_tokens ?? "ayrı raporlanmadı"}`}
              </Text>
            )}
            {model.lines.length > 0 && model.reasoning_tokens == null && (
              <Text font="secondary-body" color="text-03">
                Reasoning ayrı raporlanmadı; output toplamı kullanıldı.
              </Text>
            )}
          </div>
        ))}
      </div>
    </details>
  );
}
interface ChatResponseUsageProps {
  messageId: number;
  fallbackDuration?: number;
  initialUsage?: ResponseUsage | null;
}
export default function ChatResponseUsage({
  messageId,
  fallbackDuration,
  initialUsage,
}: ChatResponseUsageProps) {
  const { data, error, isLoading } = useSWR<ResponseUsage>(
    initialUsage && initialUsage.status !== "running"
      ? null
      : `/api/chat/message/${messageId}/usage`,
    errorHandlingFetcher,
    {
      refreshInterval: (usage) => (usage?.status === "running" ? 2000 : 0),
      revalidateOnFocus: false,
      shouldRetryOnError: false,
    }
  );
  if (error) return null;
  if (isLoading)
    return (
      <Text font="secondary-body" color="text-03">
        Süre ve maliyet yükleniyor…
      </Text>
    );
  return (
    <ResponseUsageView
      usage={data ?? initialUsage}
      fallbackDuration={fallbackDuration}
    />
  );
}
