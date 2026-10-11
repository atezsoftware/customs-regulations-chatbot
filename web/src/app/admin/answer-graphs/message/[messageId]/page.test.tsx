import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import AnswerGraphPage from "./page";

const mockUseSWR = jest.fn();

jest.mock("swr", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockUseSWR(...args),
}));

jest.mock("@/providers/UserProvider", () => ({
  useUser: () => ({ isAdmin: true, user: { id: "admin" } }),
}));

it("starts Markdown as a native URL download instead of buffering the response", async () => {
  mockUseSWR.mockImplementation((url: string | null) =>
    url?.includes("by-message")
      ? {
          data: {
            run_id: "00000000-0000-0000-0000-000000000001",
            status: "COMPLETE",
            capture_status: "COMPLETE",
            capture_error: null,
            model_name: "gpt-test",
          },
          error: undefined,
          isLoading: false,
        }
      : { data: undefined, error: undefined, isLoading: false }
  );
  const fetchMock = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ nodes: [], edges: [], next_offset: null }),
    blob: async () => new Blob(["buffered"]),
  });
  global.fetch = fetchMock;
  let downloadedHref: string | null = null;
  jest
    .spyOn(HTMLAnchorElement.prototype, "click")
    .mockImplementation(function (this: HTMLAnchorElement) {
      downloadedHref = this.getAttribute("href");
    });

  const params = Promise.resolve({ messageId: "4816" });
  await act(async () => {
    render(<AnswerGraphPage params={params} />);
    await params;
  });
  const button = await screen.findByRole("button", {
    name: "Download detailed Markdown",
  });
  await waitFor(() => expect(fetchMock).toHaveBeenCalled());
  fetchMock.mockClear();

  fireEvent.click(button);

  expect(downloadedHref).toBe(
    "/api/admin/answer-graphs/00000000-0000-0000-0000-000000000001/markdown"
  );
  expect(fetchMock).not.toHaveBeenCalled();
});
