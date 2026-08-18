import { render, screen, setupUser, waitFor } from "@tests/setup/test-utils";
import DocumentSetFiles from "@/sections/document-sets/DocumentSetFiles";

const mockUploadDocumentSetFiles = jest.fn();

jest.mock("@/app/admin/documents/sets/lib", () => ({
  __esModule: true,
  getDocumentSetFiles: jest.fn().mockResolvedValue([]),
  getDocumentSetFilesKey: (documentSetId: number) =>
    `document-set-files-${documentSetId}`,
  unlinkFileFromDocumentSet: jest.fn().mockResolvedValue(undefined),
  uploadDocumentSetFiles: (...args: unknown[]) =>
    mockUploadDocumentSetFiles(...args),
}));

jest.mock("@/lib/settings/hooks", () => ({
  useSettings: () => ({ document_import_enabled: true }),
}));

jest.mock("@/sections/document-sets/RegulatoryFileDetails", () => ({
  __esModule: true,
  default: () => null,
}));

function renderFiles() {
  return render(
    <DocumentSetFiles documentSetId={7} documentSetName="Gumruk Mevzuati" />
  );
}

beforeEach(() => {
  mockUploadDocumentSetFiles.mockReset();
  mockUploadDocumentSetFiles.mockResolvedValue({
    user_files: [],
    rejected_files: [],
  });
});

test("offers an archive upload control restricted to zip files", async () => {
  renderFiles();

  const archiveInput = await screen.findByLabelText(
    "Upload archive to document set"
  );

  expect(archiveInput).toHaveAttribute("accept", ".zip");
});

test("uploads a selected archive through the document set upload endpoint", async () => {
  const user = setupUser();
  renderFiles();

  const archiveInput = await screen.findByLabelText(
    "Upload archive to document set"
  );
  const archive = new File(
    [new Uint8Array([0x50, 0x4b, 0x03, 0x04])],
    "mevzuat.zip",
    { type: "application/zip" }
  );

  await user.upload(archiveInput, archive);

  await waitFor(() => {
    expect(mockUploadDocumentSetFiles).toHaveBeenCalledWith(7, [archive]);
  });
});
