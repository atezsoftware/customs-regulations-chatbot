import { render, screen, setupUser, waitFor } from "@tests/setup/test-utils";
import { PointerEventsCheckLevel } from "@testing-library/user-event";
import VertexAIModal from "@/sections/modals/languageModels/VertexAIModal";
import { useLLMProviders } from "@/lib/languageModels/hooks";
import type { LLMProviderView } from "@/lib/languageModels/types";

jest.mock("@/hooks/useTierAtLeast", () => ({ useTierAtLeast: () => false }));

const provider: LLMProviderView = {
  id: 7,
  name: "Gemini",
  provider: "vertex_ai",
  api_key: null,
  api_base: null,
  api_version: null,
  deployment_name: null,
  is_public: true,
  is_auto_mode: true,
  groups: [],
  personas: [],
  custom_config: {
    vertex_auth_method: "service_account_json",
    vertex_credentials: "****",
    vertex_location: "global",
  },
  model_configurations: [
    {
      id: 70,
      name: "gemini-current-pro",
      display_name: "Current Pro",
      effectiveDisplayName: "Current Pro",
      is_visible: true,
      max_input_tokens: 32000,
      supports_image_input: true,
      supports_reasoning: true,
    },
  ],
};

function OpenChatCatalog() {
  const { llmProviders } = useLLMProviders(42);
  return (
    <output data-testid="chat-models">
      {llmProviders
        ?.flatMap((item) =>
          item.model_configurations.map((model) => model.name)
        )
        .join(",")}
    </output>
  );
}

it("refreshes the open chat catalog immediately after discovering models with saved credentials", async () => {
  let savedProvider = provider;
  const user = setupUser({ pointerEventsCheck: PointerEventsCheckLevel.Never });
  const fetchSpy = jest
    .spyOn(global, "fetch")
    .mockImplementation(async (input) => {
      const url = String(input);
      let data: unknown = [];
      if (url.endsWith("/vertex-ai/available-models")) {
        const discovered = {
          name: "gemini-future-flash",
          display_name: "Future Flash",
          max_input_tokens: 1000000,
          supports_image_input: true,
          supports_reasoning: true,
        };
        savedProvider = {
          ...provider,
          model_configurations: [
            ...provider.model_configurations,
            {
              ...discovered,
              id: 71,
              is_visible: true,
              effectiveDisplayName: "Future Flash",
            },
          ],
        };
        data = [discovered];
      } else if (url.includes("/llm/persona/") || url === "/api/llm/provider") {
        data = {
          providers: [savedProvider],
          default_text: { provider_id: 7, model_name: "gemini-current-pro" },
          default_vision: null,
          default_chat_naming: null,
        };
      } else if (url.includes("/built-in/options/")) {
        data = {
          name: "vertex_ai",
          known_models: [],
          recommended_default_model: null,
        };
      } else if (url === "/api/settings") {
        data = { ee_features_enabled: false };
      } else if (url.startsWith("/api/manage/users?")) {
        data = { accepted: [], invited: [], api_keys: [] };
      }
      return { ok: true, status: 200, json: async () => data } as Response;
    });
  try {
    render(
      <>
        <OpenChatCatalog />
        <VertexAIModal existingLlmProvider={provider} />
      </>
    );
    await waitFor(() =>
      expect(screen.getByTestId("chat-models")).toHaveTextContent(
        "gemini-current-pro"
      )
    );
    await user.click(screen.getByRole("button", { name: "Refresh models" }));
    await waitFor(() =>
      expect(screen.getByTestId("chat-models")).toHaveTextContent(
        "gemini-future-flash"
      )
    );
    expect(screen.getByRole("switch")).toBeChecked();
  } finally {
    fetchSpy.mockRestore();
  }
});
