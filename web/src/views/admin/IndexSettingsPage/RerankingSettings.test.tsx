import {
  act,
  render,
  screen,
  setupUser,
  waitFor,
} from "@tests/setup/test-utils";
import { SWR_KEYS } from "@/lib/swr-keys";
import type { RerankingConfigView } from "@/lib/indexing/types";
import RerankingSettings from "@/views/admin/IndexSettingsPage/RerankingSettings";

const EMPTY_CONFIG: RerankingConfigView = {
  enabled: false,
  provider_type: null,
  model_id: null,
  api_key_configured: false,
  masked_api_key: null,
};

const STORED_CONFIG: RerankingConfigView = {
  enabled: false,
  provider_type: "siliconflow",
  model_id: "Qwen/Qwen3-Reranker-8B",
  api_key_configured: true,
  masked_api_key: "********last4",
};

const OPENROUTER_CONFIG: RerankingConfigView = {
  ...STORED_CONFIG,
  enabled: true,
  provider_type: "openrouter",
  model_id: "voyageai/rerank-3",
};

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

function renderSettings(config: RerankingConfigView = EMPTY_CONFIG) {
  return render(<RerankingSettings />, {
    swrConfig: {
      fallback: { [SWR_KEYS.rerankingConfig]: config },
      revalidateOnMount: false,
    },
  });
}

function requestBody(fetchMock: jest.Mock, callIndex = 0) {
  const init = fetchMock.mock.calls[callIndex]?.[1] as RequestInit | undefined;
  return JSON.parse(String(init?.body)) as Record<string, unknown>;
}

async function selectProvider(
  user: ReturnType<typeof setupUser>,
  label: "OpenRouter" | "SiliconFlow"
) {
  act(() =>
    screen.getByRole("combobox", { name: "Reranking provider" }).focus()
  );
  await user.keyboard("{ArrowDown}");
  await user.click(await screen.findByRole("option", { name: label }));
}

describe("RerankingSettings", () => {
  const fetchMock = jest.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    global.fetch = fetchMock;
    HTMLElement.prototype.scrollIntoView = jest.fn();
  });

  it("retains a stored masked key by omitting it from a disabled save", async () => {
    const user = setupUser();
    fetchMock.mockResolvedValueOnce(jsonResponse(STORED_CONFIG));

    renderSettings(STORED_CONFIG);

    expect(screen.getByLabelText("SiliconFlow API key")).toHaveAttribute(
      "placeholder",
      "********last4"
    );
    await user.click(
      screen.getByRole("button", { name: "Save disabled configuration" })
    );

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(requestBody(fetchMock)).toEqual({
      enabled: false,
      provider_type: "siliconflow",
      model_id: "Qwen/Qwen3-Reranker-8B",
    });
  });

  it("loads the catalog with an unsaved key without replacing a manual model ID", async () => {
    const user = setupUser();
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        models: [{ id: "Qwen/Qwen3-Reranker-4B", name: "Qwen3 Reranker 4B" }],
      })
    );

    renderSettings();

    await user.type(screen.getByLabelText("SiliconFlow API key"), "sk-unsaved");
    await user.click(screen.getByRole("button", { name: "Load models" }));

    expect(
      await screen.findByRole("combobox", {
        name: "SiliconFlow reranking model catalog",
      })
    ).toBeInTheDocument();
    expect(screen.getByLabelText("SiliconFlow model ID")).toHaveValue(
      "Qwen/Qwen3-Reranker-8B"
    );
    expect(fetchMock).toHaveBeenCalledWith(SWR_KEYS.siliconFlowRerankingModels);
  });

  it("invalidates a successful exact-configuration test when the model changes", async () => {
    const user = setupUser();
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ success: true, test_attestation: "exact-token" })
    );

    renderSettings();

    await user.type(screen.getByLabelText("SiliconFlow API key"), "sk-test");
    await user.click(screen.getByRole("switch", { name: "Enable reranking" }));
    expect(
      screen.getByRole("button", { name: "Enable and save" })
    ).toBeDisabled();

    await user.click(
      screen.getByRole("button", { name: "Test configuration" })
    );
    expect(
      await screen.findByText("Configuration test passed")
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Enable and save" })
    ).toBeEnabled();

    await user.type(screen.getByLabelText("SiliconFlow model ID"), "-new");

    expect(
      screen.getByRole("button", { name: "Enable and save" })
    ).toBeDisabled();
    expect(
      screen.queryByText("Configuration test passed")
    ).not.toBeInTheDocument();
  });

  it("saves an enabled configuration only with its exact attestation", async () => {
    const user = setupUser();
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({ success: true, test_attestation: "exact-token" })
      )
      .mockResolvedValueOnce(
        jsonResponse({
          enabled: true,
          provider_type: "siliconflow",
          model_id: "Qwen/Qwen3-Reranker-8B",
          api_key_configured: true,
          masked_api_key: "********",
        })
      );

    renderSettings();

    await user.type(screen.getByLabelText("SiliconFlow API key"), "sk-test");
    await user.click(screen.getByRole("switch", { name: "Enable reranking" }));
    await user.click(
      screen.getByRole("button", { name: "Test configuration" })
    );
    await screen.findByText("Configuration test passed");
    await user.click(screen.getByRole("button", { name: "Enable and save" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(requestBody(fetchMock, 1)).toEqual({
      enabled: true,
      provider_type: "siliconflow",
      model_id: "Qwen/Qwen3-Reranker-8B",
      api_key: "sk-test",
      test_attestation: "exact-token",
    });
  });

  it("shows backend detail when SiliconFlow rejects a test", async () => {
    const user = setupUser();
    fetchMock.mockResolvedValueOnce(
      jsonResponse(
        {
          detail: "SiliconFlow rejected this key.",
        },
        502
      )
    );

    renderSettings(STORED_CONFIG);

    await user.click(
      screen.getByRole("button", { name: "Test configuration" })
    );

    expect(
      await screen.findByText("SiliconFlow rejected this key.")
    ).toBeInTheDocument();
  });

  it("allows saving a disabled configuration without a test", async () => {
    const user = setupUser();
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        enabled: false,
        provider_type: "siliconflow",
        model_id: "Qwen/Qwen3-Reranker-8B",
        api_key_configured: true,
        masked_api_key: "********",
      })
    );

    renderSettings();

    await user.type(screen.getByLabelText("SiliconFlow API key"), "sk-new");
    const saveButton = screen.getByRole("button", {
      name: "Save disabled configuration",
    });
    expect(saveButton).toBeEnabled();
    await user.click(saveButton);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(requestBody(fetchMock)).toEqual({
      enabled: false,
      provider_type: "siliconflow",
      model_id: "Qwen/Qwen3-Reranker-8B",
      api_key: "sk-new",
    });
  });

  it("purges the persisted key only after delete confirmation", async () => {
    const user = setupUser();
    fetchMock.mockResolvedValueOnce(jsonResponse(null, 204));

    renderSettings(STORED_CONFIG);

    await user.click(
      screen.getByRole("button", { name: "Delete configuration" })
    );
    expect(
      screen.getByRole("dialog", { name: "Delete reranking configuration" })
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Delete and purge" }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        SWR_KEYS.rerankingConfig,
        expect.objectContaining({ method: "DELETE" })
      )
    );
    expect(
      await screen.findByText("Reranking configuration deleted")
    ).toBeInTheDocument();
  });

  it("discloses external query processing and the selected provider", () => {
    renderSettings();

    expect(
      screen.getByText(
        /query and authorized candidate text leave this deployment/i
      )
    ).toBeInTheDocument();
    expect(
      screen.getByText(/sent to SiliconFlow for reranking/i)
    ).toBeInTheDocument();
  });

  it("retains the persisted OpenRouter provider, model, and encrypted key on save", async () => {
    const user = setupUser();
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({ success: true, test_attestation: "openrouter-token" })
      )
      .mockResolvedValueOnce(jsonResponse(OPENROUTER_CONFIG));

    renderSettings(OPENROUTER_CONFIG);

    expect(screen.getByLabelText("OpenRouter model ID")).toHaveValue(
      "voyageai/rerank-3"
    );
    expect(screen.getByLabelText("OpenRouter API key")).toHaveAttribute(
      "placeholder",
      "********last4"
    );
    expect(screen.getByText(/sent to OpenRouter for reranking/i)).toBeVisible();
    await user.click(
      screen.getByRole("button", { name: "Test configuration" })
    );
    await screen.findByText("Configuration test passed");
    await user.click(screen.getByRole("button", { name: "Enable and save" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(requestBody(fetchMock, 0)).toEqual({
      provider_type: "openrouter",
      model_id: "voyageai/rerank-3",
    });
    expect(requestBody(fetchMock, 1)).toEqual({
      enabled: true,
      provider_type: "openrouter",
      model_id: "voyageai/rerank-3",
      test_attestation: "openrouter-token",
    });
  });

  it("requires a new provider key before testing, loading OpenRouter models, or saving a provider change", async () => {
    const user = setupUser();
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ models: [{ id: "voyageai/rerank-3", name: "Rerank 3" }] })
    );

    renderSettings(STORED_CONFIG);
    await selectProvider(user, "OpenRouter");

    expect(screen.getByLabelText("OpenRouter API key")).toHaveValue("");
    expect(screen.getByLabelText("OpenRouter API key")).toHaveAttribute(
      "placeholder",
      "Enter your OpenRouter API key"
    );
    expect(
      screen.getByRole("button", { name: "Test configuration" })
    ).toBeDisabled();
    expect(screen.getByRole("button", { name: "Load models" })).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Save disabled configuration" })
    ).toBeDisabled();
    expect(fetchMock).not.toHaveBeenCalled();

    await user.type(
      screen.getByLabelText("OpenRouter API key"),
      "sk-openrouter"
    );
    await user.click(screen.getByRole("button", { name: "Load models" }));
    await screen.findByRole("combobox", {
      name: "OpenRouter reranking model catalog",
    });

    expect(fetchMock).toHaveBeenCalledWith(
      SWR_KEYS.openRouterRerankingModels,
      expect.objectContaining({ method: "POST" })
    );
    expect(requestBody(fetchMock)).toEqual({ api_key: "sk-openrouter" });
  });

  it("clears successful test, typed key, and model catalog when the provider changes", async () => {
    const user = setupUser();
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({
          models: [{ id: "voyageai/rerank-3", name: "Rerank 3" }],
        })
      )
      .mockResolvedValueOnce(
        jsonResponse({ success: true, test_attestation: "openrouter-token" })
      );

    renderSettings(OPENROUTER_CONFIG);
    await user.type(
      screen.getByLabelText("OpenRouter API key"),
      "sk-replacement"
    );
    await user.click(screen.getByRole("button", { name: "Load models" }));
    await screen.findByRole("combobox", {
      name: "OpenRouter reranking model catalog",
    });
    await user.click(
      screen.getByRole("button", { name: "Test configuration" })
    );
    await screen.findByText("Configuration test passed");
    expect(
      screen.getByRole("button", { name: "Enable and save" })
    ).toBeEnabled();

    await selectProvider(user, "SiliconFlow");

    expect(screen.getByLabelText("SiliconFlow API key")).toHaveValue("");
    expect(screen.getByLabelText("SiliconFlow model ID")).toHaveValue(
      "Qwen/Qwen3-Reranker-8B"
    );
    expect(
      screen.queryByRole("combobox", {
        name: "OpenRouter reranking model catalog",
      })
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("Configuration test passed")
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Enable and save" })
    ).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Test configuration" })
    ).toBeDisabled();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("invalidates a successful test when the API key changes", async () => {
    const user = setupUser();
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ success: true, test_attestation: "openrouter-token" })
    );

    renderSettings(OPENROUTER_CONFIG);
    await user.click(
      screen.getByRole("button", { name: "Test configuration" })
    );
    await screen.findByText("Configuration test passed");
    await user.type(screen.getByLabelText("OpenRouter API key"), "sk-new");

    expect(
      screen.getByRole("button", { name: "Enable and save" })
    ).toBeDisabled();
  });
});
