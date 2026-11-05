import { render, screen, setupUser, waitFor } from "@tests/setup/test-utils";
import ASv3SessionAttachments from "@/sections/input/ASv3SessionAttachments";
import { ChatFileType } from "@/app/app/interfaces";

const originalFetch = global.fetch;
const file = {
  id: "saved-file-id",
  name: "inceleme.pdf",
  type: ChatFileType.PLAIN_TEXT,
  token_count: 20,
  size: 512,
  is_session_attachment: true,
};

afterEach(() => {
  global.fetch = originalFetch;
});

it("loads retained session files and removes them from future context", async () => {
  const fetchMock = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => [file],
  });
  global.fetch = fetchMock;
  render(
    <ASv3SessionAttachments
      chatSessionId="owned-session"
      draftFileIds={[]}
      revision=""
      onUseDraftFile={jest.fn()}
    />
  );

  await screen.findByText("inceleme.pdf");
  expect(fetchMock).toHaveBeenCalledWith(
    "/api/chat/sessions/owned-session/files"
  );
  expect(screen.getByRole("link")).toHaveAttribute(
    "href",
    "/api/chat/files/saved-file-id"
  );

  await setupUser().click(
    screen.getByRole("button", {
      name: "inceleme.pdf dosyasını oturumdan kaldır",
    })
  );

  await waitFor(() =>
    expect(screen.queryByText("inceleme.pdf")).not.toBeInTheDocument()
  );
  expect(fetchMock).toHaveBeenCalledWith(
    "/api/chat/sessions/owned-session/files/saved-file-id",
    { method: "DELETE" }
  );
});

it.each(["owned-session", null])(
  "does not duplicate files already visible in draft upload chips (%s)",
  async (sessionId) => {
    global.fetch = jest
      .fn()
      .mockResolvedValue({ ok: true, json: async () => [file] });
    const { rerender } = render(
      <ASv3SessionAttachments
        chatSessionId={sessionId}
        draftFileIds={[file.id]}
        revision=""
        onUseDraftFile={jest.fn()}
      />
    );
    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    expect(screen.queryByText("inceleme.pdf")).not.toBeInTheDocument();

    rerender(
      <ASv3SessionAttachments
        chatSessionId={sessionId}
        draftFileIds={[]}
        revision=""
        onUseDraftFile={jest.fn()}
      />
    );
    await screen.findByText("inceleme.pdf");
  }
);

it("recovers unbound drafts only after an explicit add action", async () => {
  const fetchMock = jest
    .fn()
    .mockResolvedValue({ ok: true, json: async () => [file] });
  global.fetch = fetchMock;
  const onUseDraftFile = jest.fn();
  render(
    <ASv3SessionAttachments
      chatSessionId={null}
      draftFileIds={[]}
      revision=""
      onUseDraftFile={onUseDraftFile}
    />
  );

  await screen.findByText("Gönderilmemiş dosyalar");
  expect(fetchMock).toHaveBeenCalledWith("/api/chat/files");
  expect(onUseDraftFile).not.toHaveBeenCalled();
  expect(screen.getByRole("link")).toHaveAttribute(
    "href",
    "/api/chat/files/saved-file-id"
  );

  await setupUser().click(
    screen.getByRole("button", { name: "Bu soruya ekle" })
  );
  expect(onUseDraftFile).toHaveBeenCalledWith(file);
});

it("deletes recovered drafts without requiring a session", async () => {
  const fetchMock = jest
    .fn()
    .mockResolvedValue({ ok: true, json: async () => [file] });
  global.fetch = fetchMock;
  render(
    <ASv3SessionAttachments
      chatSessionId={null}
      draftFileIds={[]}
      revision=""
      onUseDraftFile={jest.fn()}
    />
  );

  await screen.findByText("inceleme.pdf");
  await setupUser().click(
    screen.getByRole("button", { name: "inceleme.pdf dosyasını sil" })
  );

  await waitFor(() =>
    expect(screen.queryByText("inceleme.pdf")).not.toBeInTheDocument()
  );
  expect(fetchMock).toHaveBeenCalledWith("/api/chat/files/saved-file-id", {
    method: "DELETE",
  });
});
