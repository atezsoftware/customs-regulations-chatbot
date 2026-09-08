import { render, screen, setupUser, waitFor } from "@tests/setup/test-utils";
import LanguageModelsPage from "@/views/admin/LanguageModelsPage";
import type { LLMProviderView } from "@/lib/languageModels/types";

const mockProviders: LLMProviderView[] = [
  {
    id: 7,
    name: null,
    provider: "vertex_ai",
    api_key: null,
    api_base: null,
    api_version: null,
    custom_config: null,
    deployment_name: null,
    is_public: true,
    is_auto_mode: true,
    groups: [],
    personas: [],
    model_configurations: [
      {
        id: 70,
        name: "gemini-current-pro",
        effectiveDisplayName: "Current Pro",
        is_visible: true,
        max_input_tokens: null,
        supports_image_input: false,
        supports_reasoning: false,
      },
      {
        id: 71,
        name: "gemini-future-flash",
        effectiveDisplayName: "Future Flash",
        is_visible: true,
        max_input_tokens: null,
        supports_image_input: false,
        supports_reasoning: false,
      },
    ],
  },
];

let mockChatProviders = mockProviders;

jest.mock("next/navigation", () => ({
  useRouter: () => ({ back: jest.fn() }),
}));

jest.mock("@/lib/languageModels/hooks", () => ({
  useAdminLLMProviders: () => ({
    llmProviders: mockProviders,
    defaultText: { provider_id: 7, model_name: "gemini-current-pro" },
  }),
  useCurrentAgentLLMProviders: () => ({
    llmProviders: mockChatProviders,
    defaultText: { provider_id: 7, model_name: "gemini-current-pro" },
  }),
}));

jest.mock("@/lib/analytics/hooks", () => ({
  ...jest.requireActual("@/lib/analytics/hooks"),
  usePHFeatureFlag: () => false,
}));

jest.mock("@/views/admin/CostOverridesPanel", () => ({
  __esModule: true,
  default: () => null,
}));

describe("admin default language model", () => {
  beforeEach(() => {
    mockChatProviders = mockProviders;
    jest.spyOn(global, "fetch").mockResolvedValue({ ok: true } as Response);
  });

  afterEach(() => jest.restoreAllMocks());

  it.each([false, true])(
    "saves an unnamed provider using the admin catalog (chat catalog empty: %s)",
    async (chatCatalogEmpty) => {
      if (chatCatalogEmpty) mockChatProviders = [];
      const user = setupUser();
      render(<LanguageModelsPage />);
      await user.click(screen.getByRole("button", { name: /Default Model/ }));
      await user.click(screen.getByRole("button", { name: /Future Flash/ }));

      await waitFor(() =>
        expect(global.fetch).toHaveBeenCalledWith("/api/admin/llm/default", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            provider_id: 7,
            model_name: "gemini-future-flash",
          }),
        })
      );
    }
  );
});
