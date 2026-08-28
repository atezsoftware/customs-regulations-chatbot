import {
  analyzeAmendment,
  approveProposal,
  extractAmendmentPdf,
  extractAmendmentUrl,
  getAmendmentAnalysis,
  listAmendmentBatches,
  retryAmendmentBatch,
} from "@/lib/regulatory/amendments";
import { listBenchmarkCitationOptions } from "@/lib/regulatory/benchmark";

function jsonResponse(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    json: async () => body,
  } as Response;
}

describe("regulatory document set API contracts", () => {
  const fetchMock = jest.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    fetchMock.mockResolvedValue(jsonResponse([]));
    global.fetch = fetchMock;
  });

  it("scopes amendment analysis and history to a document set", async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({ batch: {}, proposals: [], unmatched_instructions: [] })
      )
      .mockResolvedValueOnce(jsonResponse([]));

    await analyzeAmendment(17, "Official update text");
    await listAmendmentBatches(17);

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/api/regulatory/amendments/analyze",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_set_id: 17,
          raw_text: "Official update text",
        }),
      }
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/api/regulatory/amendments/batches?document_set_id=17"
    );
  });

  it("loads benchmark citation options from a document set", async () => {
    await listBenchmarkCitationOptions(17);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/regulatory/benchmark/document-sets/17/citation-options",
      undefined
    );
  });

  it("polls and retries durable amendment batches", async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({ batch: {}, proposals: [], unmatched_instructions: [] })
      )
      .mockResolvedValueOnce(jsonResponse({ id: 42, status: "queued" }));

    await getAmendmentAnalysis(42);
    await retryAmendmentBatch(42);

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/api/regulatory/amendments/batches/42/analysis"
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/api/regulatory/amendments/batches/42/retry",
      { method: "POST" }
    );
  });

  it("sends the reviewed chunk draft with approval", async () => {
    const reviewedDraft = {
      user_file_id: "00000000-0000-0000-0000-000000000123",
      position: 15,
      text: "MADDE 15 - (2) a) (Mülga)",
      effective_start_date: "2026-08-28",
      effective_end_date: null,
    };

    await approveProposal(19, reviewedDraft);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/regulatory/amendments/proposals/19/approve",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ new_chunk_draft: reviewedDraft }),
      }
    );
  });

  it("extracts an amendment URL before analysis", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        text: "MADDE 1- Yeni metin.",
        source_type: "html",
        display_name: "20260826-2.htm",
      })
    );

    await extractAmendmentUrl("https://example.gov/20260826-2.htm");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/regulatory/amendments/sources/url",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: "https://example.gov/20260826-2.htm" }),
      }
    );
  });

  it("uploads an amendment PDF as multipart data", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        text: "MADDE 1- Yeni metin.",
        source_type: "pdf",
        display_name: "update.pdf",
      })
    );
    const file = new File(["%PDF-1.7"], "update.pdf", {
      type: "application/pdf",
    });

    await extractAmendmentPdf(file);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/regulatory/amendments/sources/pdf",
      {
        method: "POST",
        body: expect.any(FormData),
      }
    );
  });
});
