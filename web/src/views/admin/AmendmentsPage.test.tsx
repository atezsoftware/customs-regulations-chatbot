import { act, render, screen, setupUser } from "@tests/setup/test-utils";

import {
  extractAmendmentPdf,
  extractAmendmentUrl,
  listAmendmentBatches,
} from "@/lib/regulatory/amendments";
import AmendmentsPage from "@/views/admin/AmendmentsPage";

let mockDocumentSets: Array<{ id: number; name: string }> = [];

jest.mock("next/navigation", () => ({
  useRouter: () => ({ back: jest.fn(), push: jest.fn(), replace: jest.fn() }),
}));

jest.mock("@/lib/hooks/useDocumentSets", () => ({
  useDocumentSets: () => ({ documentSets: mockDocumentSets }),
}));

jest.mock("@/lib/regulatory/amendments", () => ({
  analyzeAmendment: jest.fn(),
  approveProposal: jest.fn(),
  extractAmendmentPdf: jest.fn(),
  extractAmendmentUrl: jest.fn(),
  listAmendmentBatches: jest.fn(),
  listAmendmentProposals: jest.fn(),
  rejectProposal: jest.fn(),
}));

const mockedExtractAmendmentUrl = extractAmendmentUrl as jest.MockedFunction<
  typeof extractAmendmentUrl
>;
const mockedExtractAmendmentPdf = extractAmendmentPdf as jest.MockedFunction<
  typeof extractAmendmentPdf
>;
const mockedListAmendmentBatches = listAmendmentBatches as jest.MockedFunction<
  typeof listAmendmentBatches
>;

beforeAll(() => {
  window.HTMLElement.prototype.scrollIntoView = jest.fn();
});

beforeEach(() => {
  mockDocumentSets = [{ id: 7, name: "Transit rules" }];
  mockedListAmendmentBatches.mockReset();
  mockedListAmendmentBatches.mockResolvedValue([]);
  mockedExtractAmendmentUrl.mockReset();
  mockedExtractAmendmentUrl.mockResolvedValue({
    text: "MADDE 1- Yeni metin.",
    source_type: "html",
    display_name: "20260826-2.htm",
  });
  mockedExtractAmendmentPdf.mockReset();
  mockedExtractAmendmentPdf.mockResolvedValue({
    text: "MADDE 2- PDF metni.",
    source_type: "pdf",
    display_name: "amendment.pdf",
  });
});

test("places extracted URL text in the editable amendment text area", async () => {
  const user = setupUser();
  render(<AmendmentsPage />);

  act(() => {
    screen.getByRole("combobox").focus();
  });
  await user.keyboard("{ArrowDown}");
  await screen.findByRole("option", { name: "Transit rules" });
  await user.keyboard("{Enter}");

  await user.click(screen.getByRole("button", { name: "URL" }));
  await user.type(
    screen.getByRole("textbox", { name: "Amendment source URL" }),
    "https://example.gov/20260826-2.htm"
  );
  await user.click(screen.getByRole("button", { name: "Extract" }));

  expect(await screen.findByDisplayValue("MADDE 1- Yeni metin.")).toBeVisible();
  expect(screen.getByRole("button", { name: "Analyze" })).toBeEnabled();
});

test("places extracted PDF text in the editable amendment text area", async () => {
  const user = setupUser();
  render(<AmendmentsPage />);

  act(() => {
    screen.getByRole("combobox").focus();
  });
  await user.keyboard("{ArrowDown}");
  await screen.findByRole("option", { name: "Transit rules" });
  await user.keyboard("{Enter}");

  await user.click(screen.getByRole("button", { name: "PDF" }));
  const file = new File(["pdf contents"], "amendment.pdf", {
    type: "application/pdf",
  });
  await user.upload(screen.getByLabelText("Amendment source PDF"), file);
  await user.click(screen.getByRole("button", { name: "Extract" }));

  expect(await screen.findByDisplayValue("MADDE 2- PDF metni.")).toBeVisible();
  expect(mockedExtractAmendmentPdf).toHaveBeenCalledWith(file);
  expect(screen.getByRole("button", { name: "Analyze" })).toBeEnabled();
});
