import { render, screen, setupUser, waitFor } from "@tests/setup/test-utils";
import IndexSettingsPage from "@/views/admin/IndexSettingsPage";
import { SWR_KEYS } from "@/lib/swr-keys";
import type { LLMProviderView } from "@/lib/languageModels/types";

const providers: LLMProviderView[] = [
  {
    id: 7,
    name: null,
    provider: "vertex_ai",
    api_key: null,
    api_base: null,
    api_version: null,
    custom_config: null,
    deployment_name: null,
    is_public: false,
    is_auto_mode: true,
    groups: [],
    personas: [],
    model_configurations: [
      {
        id: 70,
        name: "gemini-current",
        effectiveDisplayName: "Current Vision",
        display_name: "Current Vision",
        is_visible: true,
        max_input_tokens: null,
        supports_image_input: true,
        supports_reasoning: false,
      },
      {
        id: 71,
        name: "gemini-selected",
        effectiveDisplayName: "Selected Vision",
        display_name: "Selected Vision",
        is_visible: true,
        max_input_tokens: null,
        supports_image_input: true,
        supports_reasoning: false,
      },
    ],
  },
];

jest.mock("next/navigation", () => ({
  useRouter: () => ({ refresh: jest.fn() }),
  usePathname: () => "/admin/configuration/index-settings",
}));
jest.mock("@/lib/settings/hooks", () => ({
  useSettings: () => ({ image_extraction_and_analysis_enabled: true }),
}));
jest.mock("@/lib/indexing/hooks", () => ({
  useSecondarySearchSettings: () => ({ data: null }),
  useReindexProgress: () => ({ data: null }),
  useCurrentEmbeddingModel: () => ({ data: null, isLoading: false }),
  useCurrentSearchSettings: () => ({ data: {}, isLoading: false }),
  useContextualSetupStatus: () => ({ data: null, isLoading: false }),
  useConfiguredEmbeddingProviders: () => ({ data: [] }),
}));
jest.mock("@/lib/languageModels/hooks", () => ({
  ...jest.requireActual("@/lib/languageModels/hooks"),
  useCurrentAgentLLMProviders: () => ({ llmProviders: [], defaultText: null }),
}));
jest.mock("@/views/admin/IndexSettingsPage/RerankingSettings", () => ({
  __esModule: true,
  default: () => null,
}));

function jsonResponse(value: unknown): Response {
  return { ok: true, json: async () => value } as Response;
}

describe("admin captioning model", () => {
  afterEach(() => jest.restoreAllMocks());

  it.each([null, "Admin-only provider"])(
    "saves and refreshes the exact admin model with provider name %s",
    async (name) => {
      let selected = "gemini-current";
      const fetchSpy = jest
        .spyOn(global, "fetch")
        .mockImplementation(async (url) => {
          if (url === "/api/admin/llm/default-vision") {
            selected = "gemini-selected";
            return jsonResponse({});
          }
          if (url === SWR_KEYS.adminLlmProviders) {
            return jsonResponse({
              providers: [{ ...providers[0], name }],
              default_text: null,
              default_vision: { provider_id: 7, model_name: selected },
            });
          }
          return jsonResponse({
            providers:
              name === null
                ? [{ ...providers[0], name: "User-visible label" }]
                : [],
            default_text: null,
            default_vision: { provider_id: 7, model_name: "gemini-current" },
          });
        });
      const user = setupUser();
      render(<IndexSettingsPage />);
      await user.click(
        await screen.findByRole("button", { name: /Captioning LLM/ })
      );
      await user.click(screen.getByRole("button", { name: /Selected Vision/ }));
      await waitFor(() =>
        expect(fetchSpy).toHaveBeenCalledWith("/api/admin/llm/default-vision", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            provider_id: 7,
            model_name: "gemini-selected",
          }),
        })
      );
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: /Captioning LLM/ })
        ).toHaveTextContent("Selected Vision")
      );
      expect(
        fetchSpy.mock.calls.filter(
          ([url]) => url === SWR_KEYS.adminLlmProviders
        ).length
      ).toBeGreaterThanOrEqual(2);
      expect(
        fetchSpy.mock.calls.filter(([, init]) => init?.method === "POST")
      ).toHaveLength(1);
    }
  );
});
