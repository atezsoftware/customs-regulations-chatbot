import { ChatFileType } from "@/app/app/interfaces";
import {
  asv3SessionFileToProjectFile,
  deleteASv3SessionFile,
  uploadASv3SessionFiles,
} from "@/lib/asv3/sessionFiles";
import { projectsFileToFileDescriptor } from "@/lib/projects/utils";

const originalFetch = global.fetch;
const descriptor = {
  id: "file-record-id",
  type: ChatFileType.PLAIN_TEXT,
  name: "inceleme.pdf",
  token_count: 25,
  size: 512,
};

afterEach(() => {
  global.fetch = originalFetch;
});

it.each([null, "owned-session"])(
  "uploads directly for an ASv3 draft or session (%s)",
  async (sessionId) => {
    const result = { files: [descriptor], rejected_files: [] };
    const fetchMock = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => result,
    });
    global.fetch = fetchMock;
    const file = new File(["pdf"], "inceleme.pdf", { type: "application/pdf" });

    await expect(uploadASv3SessionFiles([file], sessionId)).resolves.toEqual(
      result
    );

    const [url, request] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/chat/files/upload");
    expect(request.method).toBe("POST");
    expect(request.body.getAll("files")).toEqual([file]);
    expect(request.body.get("chat_session_id")).toBe(sessionId);
    expect(request.headers).toBeUndefined();
  }
);

it("preserves FileRecord identity when sending a session attachment", () => {
  const attached = asv3SessionFileToProjectFile(descriptor);
  expect(attached).toMatchObject({
    token_count: 25,
    status: "COMPLETED",
    is_session_attachment: true,
  });
  expect(projectsFileToFileDescriptor(attached)).toEqual({
    id: descriptor.id,
    type: descriptor.type,
    name: descriptor.name,
    is_session_attachment: true,
  });
  expect(
    projectsFileToFileDescriptor({ ...attached, is_session_attachment: false })
  ).toMatchObject({ user_file_id: attached.id });
});

it("keeps backend upload failures visible to the caller", async () => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: false,
    json: async () => ({ detail: "Session not found" }),
  });
  await expect(uploadASv3SessionFiles([], "other-session")).rejects.toThrow(
    "Session not found"
  );
});

it("removes attachments through the session-scoped endpoint", async () => {
  const fetchMock = jest.fn().mockResolvedValue({ ok: true });
  global.fetch = fetchMock;
  await deleteASv3SessionFile("owned-session", descriptor.id);
  expect(fetchMock).toHaveBeenCalledWith(
    `/api/chat/sessions/owned-session/files/${descriptor.id}`,
    { method: "DELETE" }
  );
});

it("removes an unbound draft without requiring a session", async () => {
  const fetchMock = jest.fn().mockResolvedValue({ ok: true });
  global.fetch = fetchMock;
  await deleteASv3SessionFile(null, descriptor.id);
  expect(fetchMock).toHaveBeenCalledWith(`/api/chat/files/${descriptor.id}`, {
    method: "DELETE",
  });
});
