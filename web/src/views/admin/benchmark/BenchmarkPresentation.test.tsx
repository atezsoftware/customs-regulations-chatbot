import { modelKey } from "@/views/admin/benchmark/BenchmarkPresentation";

test("keeps identical model names from different provider rows distinct", () => {
  const firstProvider = {
    provider: "openrouter",
    provider_id: 3,
    model_id: "openai/gpt-5",
  };
  const secondProvider = {
    provider: "openrouter",
    provider_id: 9,
    model_id: "openai/gpt-5",
  };

  expect(modelKey(firstProvider)).not.toBe(modelKey(secondProvider));
});
