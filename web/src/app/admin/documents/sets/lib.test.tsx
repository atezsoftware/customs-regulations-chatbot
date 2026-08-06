import {
  getDocumentSetFiles,
  linkFileToDocumentSet,
  unlinkFileFromDocumentSet,
  uploadDocumentSetFiles,
} from "./lib";

function jsonResponse(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    json: async () => body,
  } as Response;
}

describe("document set file service", () => {
  const fetchMock = jest.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    global.fetch = fetchMock;
  });

  it("fetches files from the document set-scoped endpoint", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse([]));

    await expect(getDocumentSetFiles(42)).resolves.toEqual([]);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/manage/admin/document-set/42/files"
    );
  });

  it("uploads files as multipart data scoped to the document set", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ user_files: [], rejected_files: [] })
    );
    const file = new File(["regulation"], "regulation.txt", {
      type: "text/plain",
    });

    await uploadDocumentSetFiles(
      42,
      [file],
      new Map([["content-hash", "temporary-id"]])
    );

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, request] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/manage/admin/document-set/42/file/upload");
    expect(request.method).toBe("POST");
    const body = request.body as FormData;
    expect(body.getAll("files")).toEqual([file]);
    expect(body.get("temp_id_map")).toBe(
      JSON.stringify({ "content-hash": "temporary-id" })
    );
  });

  it("links and unlinks an encoded file id without deleting the file", async () => {
    fetchMock.mockResolvedValue(jsonResponse(null));

    await linkFileToDocumentSet(42, "file/id");
    await unlinkFileFromDocumentSet(42, "file/id");

    const endpoint = "/api/manage/admin/document-set/42/files/file%2Fid";
    expect(fetchMock).toHaveBeenNthCalledWith(1, endpoint, { method: "POST" });
    expect(fetchMock).toHaveBeenNthCalledWith(2, endpoint, {
      method: "DELETE",
    });
  });
});
