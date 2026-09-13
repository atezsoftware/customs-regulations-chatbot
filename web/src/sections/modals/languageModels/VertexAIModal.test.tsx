import {
  render,
  screen,
  setupUser,
  waitFor,
  within,
} from "@tests/setup/test-utils";
import { PointerEventsCheckLevel } from "@testing-library/user-event";
import VertexAIModal from "@/sections/modals/languageModels/VertexAIModal";
import type { LLMProviderView } from "@/lib/languageModels/types";

jest.mock("swr", () => ({
  ...jest.requireActual("swr"),
  __esModule: true,
  default: (url: string | null) => ({
    data: url?.includes("/built-in/options/vertex_ai")
      ? {
          name: "vertex_ai",
          recommended_default_model: null,
          known_models: [
            {
              name: "gemini-static-pro",
              display_name: "Static Pro",
              is_visible: true,
              max_input_tokens: 32000,
              supports_image_input: true,
              supports_reasoning: true,
            },
          ],
        }
      : undefined,
    isLoading: false,
  }),
  useSWRConfig: () => ({ mutate: jest.fn() }),
}));
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
    vertex_auth_method: "workload_identity",
    vertex_project: "my-project",
    vertex_location: "global",
  },
  model_configurations: [
    {
      name: "gemini-current-pro",
      display_name: "Current Pro",
      custom_display_name: "My Pro",
      effectiveDisplayName: "My Pro",
      is_visible: true,
      max_input_tokens: 32000,
      supports_image_input: true,
      supports_reasoning: true,
    },
  ],
};

describe("Refreshing the provider named Gemini", () => {
  afterEach(() => jest.restoreAllMocks());

  it("uses the existing Vertex credentials without exposing a separate Batch key", async () => {
    const user = setupUser({
      pointerEventsCheck: PointerEventsCheckLevel.Never,
    });
    const fetchSpy = jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      json: async () => provider,
    } as Response);
    render(
      <VertexAIModal
        existingLlmProvider={{ ...provider, has_gemini_batch_api_key: true }}
        onSuccess={jest.fn()}
      />
    );

    expect(
      screen.queryByLabelText("Gemini Batch API key")
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Remove saved Batch key" })
    ).not.toBeInTheDocument();

    const region = screen.getByPlaceholderText("global");
    await user.clear(region);
    await user.type(region, "us-central1");
    await user.click(screen.getByRole("button", { name: "Update" }));
    await waitFor(() =>
      expect(fetchSpy).toHaveBeenCalledWith(
        "/api/admin/llm/provider",
        expect.objectContaining({ method: "PUT" })
      )
    );
    const save = fetchSpy.mock.calls.find(
      ([url]) => url === "/api/admin/llm/provider"
    );
    const body = JSON.parse(save![1]!.body as string);
    expect(body.custom_config).toEqual({
      ...provider.custom_config,
      vertex_location: "us-central1",
    });
    expect(body).not.toHaveProperty("api_key");
    expect(body.api_key_changed).toBe(false);
    expect(body).not.toHaveProperty("remove_gemini_batch_api_key");
    expect(body).not.toHaveProperty("gemini_batch_api_key");
    expect(
      fetchSpy.mock.calls.some(([url]) => url === "/api/admin/llm/test")
    ).toBe(false);
  });

  it("uses the saved dynamic catalog when opening an existing provider", () => {
    render(<VertexAIModal existingLlmProvider={provider} />);
    expect(screen.getByText("My Pro")).toBeInTheDocument();
    expect(screen.queryByText("Static Pro")).not.toBeInTheDocument();
  });

  it("does not overwrite connection edits made while refresh is pending", async () => {
    const user = setupUser({
      pointerEventsCheck: PointerEventsCheckLevel.Never,
    });
    let resolveRefresh!: (value: Response) => void;
    jest.spyOn(global, "fetch").mockReturnValue(
      new Promise<Response>((resolve) => {
        resolveRefresh = resolve;
      })
    );
    render(<VertexAIModal existingLlmProvider={provider} />);
    await user.click(screen.getByRole("button", { name: "Refresh models" }));
    const region = screen.getByPlaceholderText("global");
    await user.clear(region);
    await user.type(region, "europe-west4");
    resolveRefresh({
      ok: true,
      json: async () => [
        {
          name: "gemini-future-flash",
          display_name: "Future Flash",
          max_input_tokens: 1000000,
          supports_image_input: true,
          supports_reasoning: true,
        },
      ],
    } as Response);
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Refresh models" })
      ).toBeEnabled()
    );
    expect(region).toHaveValue("europe-west4");
    expect(screen.queryByText("Future Flash")).not.toBeInTheDocument();
    expect(screen.getByText("My Pro")).toBeInTheDocument();
  });

  it.each([32000, null])(
    "preserves manual selections, names, and context limit %s",
    async (maxInputTokens) => {
      const user = setupUser({
        pointerEventsCheck: PointerEventsCheckLevel.Never,
      });
      const fetchSpy = jest
        .spyOn(global, "fetch")
        .mockImplementation(async (input) => {
          if (String(input).endsWith("/vertex-ai/available-models")) {
            return {
              ok: true,
              json: async () => [
                {
                  name: "gemini-current-pro",
                  display_name: "Current Pro",
                  max_input_tokens: 1000000,
                  supports_image_input: true,
                  supports_reasoning: true,
                },
                {
                  name: "gemini-future-flash",
                  display_name: "Future Flash",
                  max_input_tokens: 1000000,
                  supports_image_input: true,
                  supports_reasoning: true,
                },
              ],
            } as Response;
          }
          return { ok: true, json: async () => provider } as Response;
        });
      render(
        <VertexAIModal
          existingLlmProvider={{
            ...provider,
            is_auto_mode: false,
            model_configurations: [
              {
                ...provider.model_configurations[0]!,
                max_input_tokens: maxInputTokens,
              },
              {
                ...provider.model_configurations[0]!,
                name: "gemini-pinned-version",
                custom_display_name: "Pinned version",
                effectiveDisplayName: "Pinned version",
              },
              {
                ...provider.model_configurations[0]!,
                name: "gemini-hidden-version",
                is_visible: false,
                custom_display_name: "Hidden version",
                effectiveDisplayName: "Hidden version",
              },
            ],
          }}
          onSuccess={jest.fn()}
        />
      );
      await user.click(screen.getByRole("button", { name: "Refresh models" }));
      const future = await screen.findByRole("button", {
        name: /Future Flash/,
      });
      expect(within(future).getByRole("checkbox")).not.toBeChecked();
      expect(
        screen.getByRole("button", { name: /My Pro/ })
      ).toBeInTheDocument();
      await user.click(future);
      await user.click(screen.getByRole("button", { name: "Update" }));
      await waitFor(() =>
        expect(fetchSpy).toHaveBeenCalledWith(
          "/api/admin/llm/provider",
          expect.objectContaining({ method: "PUT" })
        )
      );
      const save = fetchSpy.mock.calls.find(
        ([url]) => url === "/api/admin/llm/provider"
      );
      const body = JSON.parse(save![1]!.body as string);
      expect(body.is_auto_mode).toBe(false);
      expect(body.model_configurations).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            name: "gemini-current-pro",
            is_visible: true,
            custom_display_name: "My Pro",
            max_input_tokens: maxInputTokens,
          }),
          expect.objectContaining({
            name: "gemini-future-flash",
            is_visible: true,
            max_input_tokens: 1000000,
          }),
          expect.objectContaining({
            name: "gemini-pinned-version",
            is_visible: true,
            custom_display_name: "Pinned version",
          }),
          expect.objectContaining({
            name: "gemini-hidden-version",
            is_visible: false,
            custom_display_name: "Hidden version",
          }),
        ])
      );
    }
  );

  it.each([false, true])(
    "keeps discovered models visible in Auto after refresh (toggle Auto again: %s)",
    async (toggleAutoAgain) => {
      const user = setupUser({
        pointerEventsCheck: PointerEventsCheckLevel.Never,
      });
      const fetchSpy = jest
        .spyOn(global, "fetch")
        .mockImplementation(async (input) => {
          if (String(input).endsWith("/vertex-ai/available-models")) {
            return {
              ok: true,
              json: async () => [
                {
                  name: "gemini-current-pro",
                  display_name: "Current Pro",
                  max_input_tokens: 1000000,
                  supports_image_input: true,
                  supports_reasoning: true,
                },
                {
                  name: "gemini-future-flash",
                  display_name: "Future Flash",
                  max_input_tokens: 1000000,
                  supports_image_input: true,
                  supports_reasoning: true,
                },
              ],
            } as Response;
          }
          return { ok: true, json: async () => provider } as Response;
        });
      render(
        <VertexAIModal existingLlmProvider={provider} onSuccess={jest.fn()} />
      );
      await user.click(screen.getByRole("button", { name: "Refresh models" }));
      await screen.findByText("Future Flash");
      const autoUpdate = screen.getByRole("switch");
      expect(autoUpdate).toBeChecked();
      if (toggleAutoAgain) {
        // Explicitly move through manual mode before returning to Auto.
        await user.click(autoUpdate);
        await user.click(autoUpdate);
      }
      expect(autoUpdate).toBeChecked();
      expect(screen.getByText("Future Flash")).toBeInTheDocument();
      expect(screen.getByText("My Pro")).toBeInTheDocument();
      await user.click(screen.getByRole("button", { name: "Update" }));
      await waitFor(() =>
        expect(fetchSpy).toHaveBeenCalledWith(
          "/api/admin/llm/provider",
          expect.objectContaining({ method: "PUT" })
        )
      );
      const save = fetchSpy.mock.calls.find(
        ([url]) => url === "/api/admin/llm/provider"
      );
      const body = JSON.parse(save![1]!.body as string);
      expect(body.is_auto_mode).toBe(true);
      expect(body.model_configurations).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            name: "gemini-current-pro",
            is_visible: true,
            custom_display_name: "My Pro",
          }),
          expect.objectContaining({
            name: "gemini-future-flash",
            is_visible: true,
          }),
        ])
      );
    }
  );

  it("retains existing models when Google cannot refresh the list", async () => {
    const user = setupUser({
      pointerEventsCheck: PointerEventsCheckLevel.Never,
    });
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: false,
      json: async () => ({ detail: "Google model discovery failed" }),
    } as Response);
    render(<VertexAIModal existingLlmProvider={provider} />);
    await user.click(screen.getByRole("button", { name: "Refresh models" }));
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Refresh models" })
      ).toBeEnabled()
    );
    expect(screen.getByText("My Pro")).toBeInTheDocument();
  });
});
