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
      onUploadFiles={jest.fn()}
    />
  );

  await screen.findByText("inceleme.pdf");
  expect(
    screen.getByText(
      "Dosyalar ASv3 Tek veya exp cropped seçildiğinde kullanılır."
    )
  ).toBeVisible();
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

it("does not duplicate files already visible in draft upload chips", async () => {
  const sessionId = "owned-session";
  global.fetch = jest
    .fn()
    .mockResolvedValue({ ok: true, json: async () => [file] });
  const { rerender } = render(
    <ASv3SessionAttachments
      chatSessionId={sessionId}
      draftFileIds={[file.id]}
      revision=""
      onUploadFiles={jest.fn()}
    />
  );
  await waitFor(() => expect(global.fetch).toHaveBeenCalled());
  expect(screen.queryByText("inceleme.pdf")).not.toBeInTheDocument();

  rerender(
    <ASv3SessionAttachments
      chatSessionId={sessionId}
      draftFileIds={[]}
      revision=""
      onUploadFiles={jest.fn()}
    />
  );
  await screen.findByText("inceleme.pdf");
});

it("offers a direct labeled device upload before a session exists", async () => {
  const fetchMock = jest
    .fn()
    .mockResolvedValue({ ok: true, json: async () => [file] });
  global.fetch = fetchMock;
  const onUploadFiles = jest.fn();
  const { container } = render(
    <ASv3SessionAttachments
      chatSessionId={null}
      draftFileIds={[]}
      revision=""
      onUploadFiles={onUploadFiles}
    />
  );
  const input =
    container.querySelector<HTMLInputElement>('input[type="file"]')!;
  const button = screen.getByRole("button", { name: "Dosya ekle" });
  expect(button).toBeVisible();
  expect(input).toHaveAttribute("accept", ".pdf,.docx,.xlsx,.csv,.txt,.md");
  const openPicker = jest.spyOn(input, "click");
  const user = setupUser();
  await user.click(button);
  expect(openPicker).toHaveBeenCalledTimes(1);
  const selected = new File(["data"], "inceleme.pdf", {
    type: "application/pdf",
  });
  await user.upload(input, selected);
  expect(onUploadFiles).toHaveBeenCalledWith([selected]);
});

it("never fetches or shows a recent or abandoned draft chooser", async () => {
  const fetchMock = jest
    .fn()
    .mockResolvedValue({ ok: true, json: async () => [file] });
  global.fetch = fetchMock;
  render(
    <ASv3SessionAttachments
      chatSessionId={null}
      draftFileIds={[]}
      revision=""
      onUploadFiles={jest.fn()}
    />
  );
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Dosya ekle" })).toBeVisible()
  );
  expect(fetchMock).not.toHaveBeenCalled();
  expect(screen.queryByText("inceleme.pdf")).not.toBeInTheDocument();
  expect(screen.queryByText("Recent Files")).not.toBeInTheDocument();
  expect(screen.queryByText("Gönderilmemiş dosyalar")).not.toBeInTheDocument();
});
