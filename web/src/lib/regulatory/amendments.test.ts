import {
  approveAnnexReview,
  createAmendmentSourcePackage,
  extractAmendmentDocx,
  getAnnexCapabilities,
  getAnnexEvidenceUrl,
  uploadAmendmentSourcePackage,
} from "@/lib/regulatory/amendments";

test("uploads a Word document to the DOCX extraction endpoint", async () => {
  const fetchSpy = jest.spyOn(global, "fetch").mockResolvedValue({
    ok: true,
    json: async () => ({
      text: "MADDE 2- Word metni.",
      source_type: "docx",
      display_name: "değişiklik.docx",
    }),
  } as Response);
  const file = new File(["docx contents"], "değişiklik.docx", {
    type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  });
  const result = await extractAmendmentDocx(file);

  expect(result).toEqual({
    text: "MADDE 2- Word metni.",
    source_type: "docx",
    display_name: "değişiklik.docx",
  });
  expect(fetchSpy).toHaveBeenCalledWith(
    "/api/regulatory/amendments/sources/docx",
    expect.objectContaining({
      method: "POST",
      body: expect.any(FormData),
    })
  );
  const firstRequest = fetchSpy.mock.calls[0];
  if (!firstRequest || !firstRequest[1]) {
    throw new Error("DOCX upload request options were not captured.");
  }
  expect((firstRequest[1].body as FormData).get("file")).toBe(file);

  fetchSpy.mockRestore();
});

test("treats a missing grouped-annex capability route as a legacy rollout", async () => {
  const fetchSpy = jest.spyOn(global, "fetch").mockResolvedValue({
    ok: false,
    status: 404,
  } as Response);

  await expect(getAnnexCapabilities()).resolves.toBeNull();
  expect(fetchSpy).toHaveBeenCalledWith(
    "/api/regulatory/amendments/capabilities"
  );

  fetchSpy.mockRestore();
});

test("creates immutable text source packages with the caller identity", async () => {
  const packageSnapshot = {
    id: "package-1",
    document_set_id: 7,
    status: "processing",
    asset_count: 0,
    total_bytes: 0,
    issues: [],
    manifest_sha256: null,
    assets: [],
    created_at: "2026-09-10T00:00:00Z",
    updated_at: "2026-09-10T00:00:00Z",
  } as const;
  const fetchSpy = jest.spyOn(global, "fetch").mockResolvedValue({
    ok: true,
    json: async () => packageSnapshot,
  } as Response);

  await expect(
    createAmendmentSourcePackage(7, "source-identity", {
      text: "EK-1 değiştirilmiştir.",
    })
  ).resolves.toEqual(packageSnapshot);
  expect(fetchSpy).toHaveBeenCalledWith(
    "/api/regulatory/amendments/source-packages",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({
        document_set_id: 7,
        idempotency_key: "source-identity",
        text: "EK-1 değiştirilmiştir.",
      }),
    })
  );

  fetchSpy.mockRestore();
});

test("uploads every supported annex file through the asynchronous package route", async () => {
  const fetchSpy = jest.spyOn(global, "fetch").mockResolvedValue({
    ok: true,
    json: async () => ({ status: "processing" }),
  } as Response);
  const file = new File(["sheet"], "EK-1.xlsx", {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });

  await uploadAmendmentSourcePackage(7, "upload-identity", file);

  const request = fetchSpy.mock.calls[0];
  expect(request?.[0]).toBe(
    "/api/regulatory/amendments/source-packages/upload"
  );
  const body = request?.[1]?.body as FormData;
  expect(body.get("document_set_id")).toBe("7");
  expect(body.get("idempotency_key")).toBe("upload-identity");
  expect(body.get("file")).toBe(file);

  fetchSpy.mockRestore();
});

test("sends only the frozen review hash for whole-group approval", async () => {
  const fetchSpy = jest.spyOn(global, "fetch").mockResolvedValue({
    ok: true,
    json: async () => ({ id: "review-1", status: "approving" }),
  } as Response);

  await approveAnnexReview(42, "review-1", "a".repeat(64));

  expect(fetchSpy).toHaveBeenCalledWith(
    "/api/regulatory/amendments/batches/42/annex-groups/review-1/approve",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ expected_review_sha256: "a".repeat(64) }),
    })
  );
  expect(getAnnexEvidenceUrl(42, "review-1", "evidence-1")).toBe(
    "/api/regulatory/amendments/batches/42/annex-groups/review-1/evidence/evidence-1"
  );

  fetchSpy.mockRestore();
});
