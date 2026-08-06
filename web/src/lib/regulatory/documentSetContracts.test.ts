import {
  analyzeAmendment,
  listAmendmentBatches,
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
});
