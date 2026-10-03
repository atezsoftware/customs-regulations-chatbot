import { render, screen } from "@testing-library/react";
import {
  ResponseUsageView,
  costLabel,
  formatResponseDuration,
} from "@/sections/chat/usage/ResponseUsage";
import type { ResponseUsage } from "@/sections/chat/usage/interfaces";

const usage: ResponseUsage = {
  duration_seconds: 847.778,
  status: "complete",
  total_cost_usd: 0.00346,
  known_cost_usd: 0.00346,
  currency: "USD",
  calls: 4,
  unpriced_calls: 0,
  models: [
    {
      model: "model",
      provider: "provider",
      source: "model_registry",
      priced_at: "2026-10-03T10:00:00Z",
      input_tokens: 1000,
      output_tokens: 200,
      reasoning_tokens: 120,
      complete: true,
      known_cost_usd: 0.00346,
      lines: [
        {
          category: "reasoning",
          tokens: 120,
          usd_per_million: 10,
          cost_usd: 0.0012,
        },
      ],
    },
  ],
};

it("shows elapsed minutes and seconds and token-rate calculations", () => {
  render(<ResponseUsageView usage={usage} />);
  expect(screen.getByText(/14 dk 8 sn/)).toHaveTextContent("$0.00346");
  expect(screen.getByText(/Reasoning: 120 token/)).toHaveTextContent(
    "$10.0000 / 1M = $0.0012"
  );
});

it("does not present missing pricing as a free answer", () => {
  expect(
    costLabel({ ...usage, status: "partial", total_cost_usd: null })
  ).toMatch(/En az.*maliyet eksik/);
  expect(
    costLabel({
      ...usage,
      status: "unavailable",
      total_cost_usd: null,
      known_cost_usd: null,
    })
  ).toBe("Maliyet hesaplanamadı");
  expect(costLabel({ ...usage, total_cost_usd: 0 })).toContain("$0.0000");
  expect(formatResponseDuration(59.9)).toBe("1 dk 0 sn");
  expect(formatResponseDuration(null)).toBe("Süre kaydı yok");
});
