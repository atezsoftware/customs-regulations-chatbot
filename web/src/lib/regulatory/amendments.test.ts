import { extractAmendmentDocx } from "@/lib/regulatory/amendments";

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
