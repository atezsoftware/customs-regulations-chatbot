import { render, screen, setupUser } from "@tests/setup/test-utils";
import { Button } from "@opal/components";
import FilePickerPopover from "@/refresh-components/popovers/FilePickerPopover";

const mockProjectsContext = {
  allRecentFiles: [],
  deleteUserFile: jest.fn(),
  setCurrentMessageFiles: jest.fn(),
};

jest.mock("@/lib/settings/hooks", () => ({
  useSettings: () => ({ document_import_enabled: false }),
}));
jest.mock("@/providers/ProjectsContext", () => ({
  useProjectsContext: () => mockProjectsContext,
}));
jest.mock("@/sections/modals/UserFilesModal", () => ({
  __esModule: true,
  default: () => null,
}));

it("offers session upload when the separate catalog importer is disabled", async () => {
  const { container } = render(
    <FilePickerPopover
      sessionAttachmentsOnly
      handleUploadChange={jest.fn()}
      trigger={<Button>Attach</Button>}
    />
  );
  expect(container.querySelector('input[type="file"]')).toHaveAttribute(
    "accept",
    ".pdf,.docx,.xlsx,.csv,.txt,.md"
  );
  await setupUser().click(screen.getByRole("button", { name: "Attach" }));
  expect(
    await screen.findByRole("button", { name: /Upload Files/ })
  ).toBeVisible();
});

it("preserves the disabled catalog importer behavior outside session mode", async () => {
  const { container } = render(
    <FilePickerPopover
      handleUploadChange={jest.fn()}
      trigger={<Button>Attach</Button>}
    />
  );
  expect(container.querySelector('input[type="file"]')).toBeNull();
  await setupUser().click(screen.getByRole("button", { name: "Attach" }));
  expect(screen.queryByText("Upload Files")).not.toBeInTheDocument();
});
