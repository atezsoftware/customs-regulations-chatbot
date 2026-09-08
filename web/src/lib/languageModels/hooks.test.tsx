import type { ReactNode } from "react";
import useSWR, { SWRConfig, useSWRConfig } from "swr";
import { act, renderHook, waitFor } from "@testing-library/react";
import { setDocumentVisibility } from "@tests/setup/test-utils";
import { errorHandlingFetcher } from "@/lib/fetcher";
import { SWR_KEYS } from "@/lib/swr-keys";
import { useLlmManager } from "@/lib/hooks";
import useMultiModelChat from "@/hooks/useMultiModelChat";
import { refreshLlmProviderCaches } from "@/lib/languageModels/cache";
import {
  useAdminLLMProviders,
  useLLMProviders,
} from "@/lib/languageModels/hooks";
import type {
  LLMProviderResponse,
  LLMProviderView,
} from "@/lib/languageModels/types";
import type { SelectedModel } from "@/sections/model-selector/MultiModelSelector";
import { ChatSession, ChatSessionSharedStatus } from "@/app/app/interfaces";
import { User, UserRole } from "@/lib/types";

let mockUser: User | null = null;
jest.mock("@/providers/UserProvider", () => {
  const actual = jest.requireActual<typeof import("@/providers/UserProvider")>(
    "@/providers/UserProvider"
  );
  return {
    ...actual,
    useUser: () => ({ ...actual.useUser(), user: mockUser }),
  };
});

function Wrapper({ children }: { children: ReactNode }) {
  return (
    <SWRConfig value={{ provider: () => new Map(), shouldRetryOnError: false }}>
      {children}
    </SWRConfig>
  );
}

const initialResponse: LLMProviderResponse<LLMProviderView> = {
  providers: [
    {
      id: 7,
      name: "Gemini",
      provider: "vertex_ai",
      api_key: null,
      api_base: null,
      api_version: null,
      custom_config: null,
      is_public: true,
      is_auto_mode: true,
      groups: [],
      personas: [],
      deployment_name: null,
      model_configurations: [
        {
          id: 70,
          name: "gemini-current-pro",
          display_name: "Current Pro",
          effectiveDisplayName: "Current Pro",
          is_visible: true,
          max_input_tokens: 32000,
          supports_image_input: false,
          supports_reasoning: false,
        },
      ],
    },
  ],
  default_text: { provider_id: 7, model_name: "gemini-current-pro" },
  default_vision: null,
  default_chat_naming: null,
};

const updatedResponse: LLMProviderResponse<LLMProviderView> = {
  ...initialResponse,
  providers: [
    {
      ...initialResponse.providers[0]!,
      model_configurations: [
        {
          ...initialResponse.providers[0]!.model_configurations[0]!,
          custom_display_name: "My renamed Pro",
          max_input_tokens: 1000000,
          supports_image_input: true,
        },
        {
          id: 71,
          name: "gemini-future-flash",
          display_name: "Future Flash",
          effectiveDisplayName: "Future Flash",
          is_visible: true,
          max_input_tokens: 1000000,
          supports_image_input: true,
          supports_reasoning: true,
        },
      ],
    },
  ],
  default_text: { provider_id: 7, model_name: "gemini-future-flash" },
  default_vision: { provider_id: 7, model_name: "gemini-current-pro" },
};

describe("provider catalog synchronization", () => {
  let response = initialResponse;

  beforeEach(() => {
    jest.useFakeTimers();
    mockUser = null;
    response = initialResponse;
    jest.spyOn(global, "fetch").mockImplementation(
      async () =>
        ({
          ok: true,
          status: 200,
          json: async () => response,
        }) as Response
    );
  });

  afterEach(() => {
    setDocumentVisibility(true);
    window.dispatchEvent(new Event("online"));
    jest.restoreAllMocks();
    jest.useRealTimers();
  });

  it.each(["public", "persona", "admin"] as const)(
    "updates an open %s consumer within a minute",
    async (consumer) => {
      const { result } = renderHook(
        () => {
          const publicProviders = useLLMProviders();
          const personaProviders = useLLMProviders(42);
          const adminProviders = useAdminLLMProviders();
          return {
            public: publicProviders,
            persona: personaProviders,
            admin: adminProviders,
          }[consumer];
        },
        { wrapper: Wrapper }
      );
      await waitFor(() => expect(result.current.llmProviders).toHaveLength(1));
      response = updatedResponse;
      await act(() => jest.advanceTimersByTimeAsync(60000));
      expect(result.current.llmProviders?.[0]?.model_configurations).toEqual([
        expect.objectContaining({
          name: "gemini-current-pro",
          effectiveDisplayName: "My renamed Pro",
          max_input_tokens: 1000000,
          supports_image_input: true,
        }),
        expect.objectContaining({
          name: "gemini-future-flash",
          is_visible: true,
        }),
      ]);
      expect(result.current.defaultText).toEqual({
        provider_id: 7,
        model_name: "gemini-future-flash",
      });
      expect(result.current.defaultVision).toEqual({
        provider_id: 7,
        model_name: "gemini-current-pro",
      });
    }
  );

  it.each(["focus", "reconnect"] as const)(
    "refreshes stale public and admin catalogs on %s",
    async (trigger) => {
      const { result } = renderHook(
        () => ({ public: useLLMProviders(), admin: useAdminLLMProviders() }),
        { wrapper: Wrapper }
      );
      await waitFor(() =>
        expect(result.current.admin.llmProviders).toHaveLength(1)
      );
      setDocumentVisibility(false);
      window.dispatchEvent(new Event("offline"));
      response = updatedResponse;
      await act(() => jest.advanceTimersByTimeAsync(60000));
      expect(result.current.public.defaultText?.model_name).toBe(
        "gemini-current-pro"
      );
      expect(result.current.admin.defaultText?.model_name).toBe(
        "gemini-current-pro"
      );
      await act(async () => {
        if (trigger === "focus") {
          // Reconnect while hidden, then foreground the tab.
          window.dispatchEvent(new Event("online"));
          await jest.advanceTimersByTimeAsync(1);
          setDocumentVisibility(true);
          window.dispatchEvent(new Event("focus"));
        } else {
          setDocumentVisibility(true);
          window.dispatchEvent(new Event("online"));
        }
        await jest.advanceTimersByTimeAsync(1);
      });
      expect(result.current.public.defaultText?.model_name).toBe(
        "gemini-future-flash"
      );
      expect(result.current.admin.defaultText?.model_name).toBe(
        "gemini-future-flash"
      );
    }
  );

  it("invalidates every mounted provider catalog after an admin change", async () => {
    const { result } = renderHook(
      () => ({
        public: useLLMProviders(),
        persona: useLLMProviders(42),
        otherPersona: useLLMProviders(99),
        admin: useAdminLLMProviders(),
        images: useSWR<LLMProviderResponse<LLMProviderView>>(
          SWR_KEYS.llmProvidersWithImageGen,
          errorHandlingFetcher
        ),
        mutate: useSWRConfig().mutate,
      }),
      { wrapper: Wrapper }
    );
    await waitFor(() => expect(result.current.images.data).toBeDefined());
    response = updatedResponse;
    await act(() => refreshLlmProviderCaches(result.current.mutate));
    for (const consumer of [
      result.current.public,
      result.current.persona,
      result.current.otherPersona,
      result.current.admin,
    ]) {
      expect(consumer.defaultText?.model_name).toBe("gemini-future-flash");
    }
    expect(result.current.images.data?.default_vision).toEqual({
      provider_id: 7,
      model_name: "gemini-current-pro",
    });
  });

  it("updates the default in an open chat without replacing a valid manual choice", async () => {
    const { result } = renderHook(() => useLlmManager(), { wrapper: Wrapper });
    await waitFor(() =>
      expect(result.current.currentLlm.modelName).toBe("gemini-current-pro")
    );
    response = updatedResponse;
    await act(() => jest.advanceTimersByTimeAsync(60000));
    expect(result.current.currentLlm.modelName).toBe("gemini-future-flash");
    act(() =>
      result.current.updateCurrentLlm({
        name: "Gemini",
        provider: "vertex_ai",
        modelName: "gemini-current-pro",
      })
    );
    await act(() => jest.advanceTimersByTimeAsync(60000));
    expect(result.current.currentLlm.modelName).toBe("gemini-current-pro");
  });

  it("replaces a removed manual model with the refreshed default", async () => {
    const { result } = renderHook(() => useLlmManager(), { wrapper: Wrapper });
    await waitFor(() =>
      expect(result.current.currentLlm.modelName).toBe("gemini-current-pro")
    );
    act(() =>
      result.current.updateCurrentLlm({
        name: "Gemini",
        provider: "vertex_ai",
        modelName: "gemini-current-pro",
      })
    );
    response = {
      ...updatedResponse,
      providers: [
        {
          ...updatedResponse.providers[0]!,
          model_configurations: [
            updatedResponse.providers[0]!.model_configurations[1]!,
          ],
        },
      ],
    };
    await act(() => jest.advanceTimersByTimeAsync(60000));
    expect(result.current.currentLlm).toEqual({
      name: "Gemini",
      provider: "vertex_ai",
      modelName: "gemini-future-flash",
    });
  });

  it("updates custom names in a multi-model selection after catalog refresh", async () => {
    response = {
      ...updatedResponse,
      providers: [
        {
          ...updatedResponse.providers[0]!,
          model_configurations: [
            initialResponse.providers[0]!.model_configurations[0]!,
            updatedResponse.providers[0]!.model_configurations[1]!,
          ],
        },
      ],
      default_text: initialResponse.default_text,
    };
    const { result } = renderHook(() => useMultiModelChat(useLlmManager()), {
      wrapper: Wrapper,
    });
    await waitFor(() =>
      expect(result.current.selectedModels[0]?.displayName).toBe("Current Pro")
    );
    act(() =>
      result.current.addModel({
        name: "Gemini",
        provider: "vertex_ai",
        modelName: "gemini-future-flash",
        modelConfigurationId: 71,
        displayName: "Future Flash",
      })
    );
    response = updatedResponse;
    await act(() => jest.advanceTimersByTimeAsync(60000));
    expect(
      result.current.selectedModels.map((model) => model.displayName)
    ).toEqual(["My renamed Pro", "Future Flash"]);
    expect(result.current.buildLlmOverrides()[0]?.display_name).toBe(
      "My renamed Pro"
    );
  });

  it.each(["remove", "replace", "add"] as const)(
    "uses reconciled comparison indices for %s after a model disappears",
    async (action) => {
      const lite: SelectedModel = {
        name: "Gemini",
        provider: "vertex_ai",
        modelName: "gemini-lite",
        modelConfigurationId: 72,
        displayName: "Lite",
      };
      const backup: SelectedModel = {
        name: "Gemini",
        provider: "vertex_ai",
        modelName: "gemini-backup",
        modelConfigurationId: 73,
        displayName: "Backup",
      };
      const catalog = [
        ...updatedResponse.providers[0]!.model_configurations,
        {
          ...initialResponse.providers[0]!.model_configurations[0]!,
          id: 72,
          name: "gemini-lite",
          display_name: "Lite",
        },
        {
          ...initialResponse.providers[0]!.model_configurations[0]!,
          id: 73,
          name: "gemini-backup",
          display_name: "Backup",
        },
      ];
      response = {
        ...initialResponse,
        providers: [
          { ...initialResponse.providers[0]!, model_configurations: catalog },
        ],
      };
      const { result } = renderHook(() => useMultiModelChat(useLlmManager()), {
        wrapper: Wrapper,
      });
      await waitFor(() =>
        expect(result.current.selectedModels).toHaveLength(1)
      );
      act(() =>
        result.current.addModel({
          name: "Gemini",
          provider: "vertex_ai",
          modelName: "gemini-future-flash",
          modelConfigurationId: 71,
          displayName: "Future Flash",
        })
      );
      act(() => result.current.addModel(lite));
      response = {
        ...initialResponse,
        providers: [
          {
            ...initialResponse.providers[0]!,
            model_configurations: catalog.filter((model) => model.id !== 71),
          },
        ],
      };
      await act(() => jest.advanceTimersByTimeAsync(60000));
      expect(
        result.current.selectedModels.map((model) => model.modelName)
      ).toEqual(["gemini-current-pro", "gemini-lite"]);
      expect(result.current.isMultiModelActive).toBe(true);
      act(() => {
        if (action === "remove") result.current.removeModel(1);
        if (action === "replace") result.current.replaceModel(1, backup);
        if (action === "add") result.current.addModel(backup);
      });
      const expected = {
        remove: ["gemini-current-pro"],
        replace: ["gemini-current-pro", "gemini-backup"],
        add: ["gemini-current-pro", "gemini-lite", "gemini-backup"],
      }[action];
      expect(
        result.current.selectedModels.map((model) => model.modelName)
      ).toEqual(expected);
      expect(
        result.current.buildLlmOverrides().map((model) => model.model_version)
      ).toEqual(expected);
    }
  );

  it("keeps the surviving comparison model after provider deletion and clears an empty catalog", async () => {
    const survivor = {
      ...initialResponse.providers[0]!,
      id: 8,
      name: "Other Gemini",
      model_configurations: [
        { ...updatedResponse.providers[0]!.model_configurations[1]!, id: 81 },
        {
          ...initialResponse.providers[0]!.model_configurations[0]!,
          id: 82,
          name: "gemini-backup",
          display_name: "Backup",
        },
      ],
    };
    response = {
      ...initialResponse,
      providers: [...initialResponse.providers, survivor],
    };
    const { result } = renderHook(
      () => {
        const manager = useLlmManager();
        return { manager, comparison: useMultiModelChat(manager) };
      },
      { wrapper: Wrapper }
    );
    await waitFor(() =>
      expect(result.current.comparison.selectedModels).toHaveLength(1)
    );
    act(() =>
      result.current.comparison.addModel({
        name: "Other Gemini",
        provider: "vertex_ai",
        modelName: "gemini-future-flash",
        modelConfigurationId: 81,
        displayName: "Future Flash",
      })
    );
    response = {
      ...initialResponse,
      providers: [survivor],
      default_text: { provider_id: 8, model_name: "gemini-backup" },
    };
    await act(() => jest.advanceTimersByTimeAsync(60000));
    expect(result.current.comparison.isMultiModelActive).toBe(false);
    expect(
      result.current.comparison.selectedModels.map((model) => model.modelName)
    ).toEqual(["gemini-future-flash"]);
    expect(result.current.manager.currentLlm.modelName).toBe(
      "gemini-future-flash"
    );
    expect(result.current.comparison.buildLlmOverrides()).toEqual([
      {
        model_provider: "Other Gemini",
        model_provider_type: "vertex_ai",
        model_version: "gemini-future-flash",
        display_name: "Future Flash",
      },
    ]);
    response = { ...initialResponse, providers: [], default_text: null };
    await act(() => jest.advanceTimersByTimeAsync(60000));
    expect(result.current.comparison.selectedModels).toEqual([]);
    expect(result.current.comparison.buildLlmOverrides()).toEqual([]);
    expect(result.current.comparison.isMultiModelActive).toBe(false);
  });

  it.each([false, true])(
    "revalidates a hidden manual model while preserving a configured default: %s",
    async (isDefault) => {
      const { result } = renderHook(() => useLlmManager(), {
        wrapper: Wrapper,
      });
      await waitFor(() =>
        expect(result.current.currentLlm.modelName).toBe("gemini-current-pro")
      );
      act(() =>
        result.current.updateCurrentLlm({
          name: "Gemini",
          provider: "vertex_ai",
          modelName: "gemini-current-pro",
        })
      );
      response = {
        ...updatedResponse,
        providers: [
          {
            ...updatedResponse.providers[0]!,
            model_configurations: [
              {
                ...updatedResponse.providers[0]!.model_configurations[0]!,
                is_visible: false,
              },
              updatedResponse.providers[0]!.model_configurations[1]!,
            ],
          },
        ],
        default_text: isDefault
          ? initialResponse.default_text
          : updatedResponse.default_text,
      };
      await act(() => jest.advanceTimersByTimeAsync(60000));
      expect(result.current.currentLlm.modelName).toBe(
        isDefault ? "gemini-current-pro" : "gemini-future-flash"
      );
    }
  );

  it.each(["saved chat", "personal default"] as const)(
    "revalidates an existing %s selection when its model is hidden",
    async (selectionSource) => {
      response = updatedResponse;
      const selection = "Gemini__vertex_ai__gemini-current-pro";
      const session: ChatSession = {
        id: "saved-chat",
        name: "Saved chat",
        persona_id: 0,
        project_id: null,
        time_created: "2026-09-08T00:00:00Z",
        time_updated: "2026-09-08T00:00:00Z",
        shared_status: ChatSessionSharedStatus.Private,
        current_alternate_model: selection,
        current_temperature_override: null,
        current_reasoning_effort_override: null,
      };
      if (selectionSource === "personal default") {
        mockUser = {
          id: "test-user",
          email: "test@example.com",
          is_active: true,
          is_superuser: false,
          is_verified: true,
          role: UserRole.BASIC,
          team_name: null,
          preferences: {
            chosen_assistants: null,
            visible_assistants: [],
            hidden_assistants: [],
            recent_assistants: [],
            auto_scroll: true,
            shortcut_enabled: false,
            temperature_override_enabled: false,
            theme_preference: null,
            chat_background: null,
            default_app_mode: "CHAT",
            default_model: selection,
          },
        };
      }
      const { result } = renderHook(
        () =>
          useLlmManager(selectionSource === "saved chat" ? session : undefined),
        { wrapper: Wrapper }
      );
      await waitFor(() =>
        expect(result.current.currentLlm.modelName).toBe("gemini-current-pro")
      );
      response = {
        ...updatedResponse,
        providers: [
          {
            ...updatedResponse.providers[0]!,
            model_configurations: [
              {
                ...updatedResponse.providers[0]!.model_configurations[0]!,
                is_visible: false,
              },
              updatedResponse.providers[0]!.model_configurations[1]!,
            ],
          },
        ],
      };
      await act(() => jest.advanceTimersByTimeAsync(60000));
      expect(result.current.currentLlm.modelName).toBe("gemini-future-flash");
    }
  );

  it("drops a hidden comparison model that shares the current model name on another provider", async () => {
    const secondProvider = {
      ...initialResponse.providers[0]!,
      id: 8,
      name: "Second Gemini",
      model_configurations: [
        { ...initialResponse.providers[0]!.model_configurations[0]!, id: 80 },
      ],
    };
    response = {
      ...initialResponse,
      providers: [...initialResponse.providers, secondProvider],
    };
    const { result } = renderHook(() => useMultiModelChat(useLlmManager()), {
      wrapper: Wrapper,
    });
    await waitFor(() => expect(result.current.selectedModels).toHaveLength(1));
    act(() =>
      result.current.addModel({
        name: "Second Gemini",
        provider: "vertex_ai",
        modelName: "gemini-current-pro",
        modelConfigurationId: 80,
        displayName: "Current Pro",
      })
    );
    expect(result.current.selectedModels).toHaveLength(2);
    response = {
      ...initialResponse,
      providers: [
        ...initialResponse.providers,
        {
          ...secondProvider,
          model_configurations: [
            { ...secondProvider.model_configurations[0]!, is_visible: false },
          ],
        },
      ],
    };
    await act(() => jest.advanceTimersByTimeAsync(60000));
    expect(
      result.current.selectedModels.map((model) => model.modelConfigurationId)
    ).toEqual([70]);
    expect(result.current.isMultiModelActive).toBe(false);
    expect(result.current.buildLlmOverrides()).toEqual([
      {
        model_provider: "Gemini",
        model_provider_type: "vertex_ai",
        model_version: "gemini-current-pro",
        display_name: "Current Pro",
      },
    ]);
  });
});
