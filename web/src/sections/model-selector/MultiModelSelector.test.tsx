import { cleanup, render, screen, within } from "@testing-library/react";
import { TooltipProvider } from "@radix-ui/react-tooltip";
import MultiModelSelector, {
  type SelectedModel,
} from "@/sections/model-selector/MultiModelSelector";

jest.mock("@/lib/settings/hooks", () => ({
  useSettings: () => ({ multi_model_chat_enabled: true }),
}));
jest.mock("@/lib/languageModels/hooks", () => ({
  useCurrentAgentLLMProviders: () => ({ llmProviders: [], isLoading: true }),
}));
jest.mock("@/sections/model-selector/ModelSelectorContent", () => ({
  __esModule: true,
  default: () => null,
  useModelDetailManagers: () => ({}),
}));

const model: SelectedModel = {
  name: "selected-provider",
  provider: "openai",
  modelName: "gpt-5-mini",
  modelConfigurationId: 1,
  displayName: "Selected model",
};

afterEach(cleanup);

it("keeps the selected model replaceable without allowing a second Supersearch model", () => {
  render(
    <TooltipProvider>
      <MultiModelSelector
        selectedModels={[model]}
        onAdd={jest.fn()}
        onRemove={jest.fn()}
        onReplace={jest.fn()}
        maxModels={1}
      />
    </TooltipProvider>
  );
  const selector = within(screen.getByTestId("model-selector"));
  expect(selector.getAllByRole("button")).toHaveLength(1);
  expect(
    selector.getByRole("button", { name: "Selected model" })
  ).toBeEnabled();
});

it("continues allowing additional models in the default workflow", () => {
  render(
    <TooltipProvider>
      <MultiModelSelector
        selectedModels={[model]}
        onAdd={jest.fn()}
        onRemove={jest.fn()}
        onReplace={jest.fn()}
      />
    </TooltipProvider>
  );
  expect(
    within(screen.getByTestId("model-selector")).getAllByRole("button")
  ).toHaveLength(2);
});
