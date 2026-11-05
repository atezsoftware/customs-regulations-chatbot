import { render, screen, setupUser } from "@tests/setup/test-utils";
import AppInputBar from "@/sections/input/AppInputBar";
import type { AppInputBarProps } from "@/sections/input/AppInputBar";

let mockAppMode = "chat";
let mockIsAdmin = false;
const mockFiles: never[] = [];
const mockProjects = {
  currentMessageFiles: mockFiles,
  currentProjectId: null,
  allRecentFiles: mockFiles,
  setCurrentMessageFiles: jest.fn(),
};
const mockShortcuts = { activePromptShortcuts: [] };
const mockDraft = {
  draft: null,
  loaded: true,
  save: jest.fn(),
  clear: jest.fn(),
};
const mockVoice = {
  stopTTS: jest.fn(),
  isTTSPlaying: false,
  isManualTTSPlaying: false,
  isTTSLoading: false,
  isAwaitingAutoPlaybackStart: false,
  isTTSMuted: false,
  toggleTTSMute: jest.fn(),
};

jest.mock("@/hooks/useAppFocus", () => ({
  __esModule: true,
  default: () => ({
    isNewSession: () => true,
    isChat: () => false,
    getId: () => null,
  }),
}));
jest.mock("@/providers/UserProvider", () => ({
  useUser: () => ({ user: null, isAdmin: mockIsAdmin }),
}));
jest.mock("@/providers/QueryControllerProvider", () => ({
  useQueryController: () => ({
    state: { phase: "idle", appMode: mockAppMode },
  }),
}));
jest.mock("@/providers/VoiceModeProvider", () => ({
  useVoiceMode: () => mockVoice,
}));
jest.mock("@/hooks/useVoiceStatus", () => ({
  useVoiceStatus: () => ({ sttEnabled: false }),
}));
jest.mock("@/providers/ProjectsContext", () => ({
  useProjectsContext: () => mockProjects,
}));
jest.mock("@/lib/hooks/useForcedTools", () => ({
  useForcedTools: () => ({ forcedToolIds: [], setForcedToolIds: jest.fn() }),
}));
jest.mock("@/hooks/usePromptShortcuts", () => ({
  __esModule: true,
  default: () => mockShortcuts,
}));
jest.mock("@/hooks/useCCPairs", () => ({
  __esModule: true,
  default: () => ({ ccPairs: [], isLoading: false }),
}));
jest.mock("@/lib/hooks", () => ({
  useFederatedConnectors: () => ({ data: [], isLoading: false }),
}));
jest.mock("@/hooks/useDraft", () => ({
  useDraft: () => mockDraft,
  draftKey: () => "new-chat",
}));
jest.mock("@/lib/settings/hooks", () => ({
  useSettings: () => ({
    document_import_enabled: false,
    vectorDbEnabled: false,
  }),
}));

const props: AppInputBarProps = {
  stopGenerating: jest.fn(),
  onSubmit: jest.fn(),
  handleFileUpload: jest.fn(),
  chatState: "input",
  currentSessionFileTokenCount: 0,
  availableContextTokens: 10000,
  deepResearchEnabled: false,
  atezSearchV3Enabled: false,
  asv3ExpCroppedEnabled: false,
  toggleDeepResearch: jest.fn(),
  disabled: false,
  selectedAgent: {
    id: 0,
    name: "Assistant",
    description: "",
    tools: [],
    starter_messages: null,
    document_sets: [],
    is_public: true,
    is_listed: true,
    display_priority: null,
    is_featured: false,
    builtin_persona: true,
    owner: null,
    owner_group: null,
    user_permission: null,
  },
  llmManager: {
    currentLlm: { name: "provider", provider: "openai", modelName: "model" },
    defaultText: null,
    updateCurrentLlm: jest.fn(),
    temperature: 1,
    updateTemperature: jest.fn(),
    temperatureExplicitlySet: false,
    reasoningEffort: null,
    updateReasoningEffort: jest.fn(),
    hasBoundSession: false,
    persistOverrides: async () => {},
    updateModelOverrideBasedOnChatSession: jest.fn(),
    imageFilesPresent: false,
    updateImageFilesPresent: jest.fn(),
    liveAgent: null,
    maxTemperature: 2,
    llmProviders: [],
    isLoadingProviders: false,
    hasAnyProvider: true,
  },
  filterManager: {
    timeRange: null,
    selectedSources: [],
    selectedDocumentSets: [],
    selectedTags: [],
    setTimeRange: jest.fn(),
    setSelectedSources: jest.fn(),
    setSelectedDocumentSets: jest.fn(),
    setSelectedTags: jest.fn(),
    getFilterString: () => "",
    buildFiltersFromQueryString: jest.fn(),
    clearFilters: jest.fn(),
  },
};

it.each([
  ["chat", false],
  ["search", false],
  ["chat", true],
] as const)(
  "shows device upload before mode selection (%s, admin %s)",
  async (appMode, isAdmin) => {
    mockAppMode = appMode;
    mockIsAdmin = isAdmin;
    const onUpload = jest.fn();
    const { container } = render(
      <AppInputBar {...props} handleFileUpload={onUpload} />
    );
    expect(screen.getByRole("button", { name: "Dosya ekle" })).toBeVisible();
    expect(screen.queryByText("Recent Files")).not.toBeInTheDocument();
    expect(
      screen.queryByText("Gönderilmemiş dosyalar")
    ).not.toBeInTheDocument();
    const input =
      container.querySelector<HTMLInputElement>('input[type="file"]')!;
    const file = new File(["data"], "inceleme.pdf", {
      type: "application/pdf",
    });
    await setupUser().upload(input, file);
    expect(onUpload).toHaveBeenCalledWith([file]);
  }
);
