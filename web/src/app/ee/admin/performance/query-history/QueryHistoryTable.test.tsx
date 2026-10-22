import {
  act,
  deferred,
  render,
  screen,
  setupUser,
  waitFor,
} from "@tests/setup/test-utils";
import { QueryHistoryTable } from "@/app/ee/admin/performance/query-history/QueryHistoryTable";
import { QueryHistoryType } from "@/lib/settings/types";

let mockSearch = "";
let mockHistoryType = QueryHistoryType.NORMAL;
const mockReplace = jest.fn();
const mockPush = jest.fn();
const mockRouter = { replace: mockReplace, push: mockPush };

jest.mock("next/navigation", () => ({
  useRouter: () => mockRouter,
  usePathname: () => "/admin/performance/query-history",
  useSearchParams: () => new URLSearchParams(mockSearch),
}));

jest.mock("@/lib/settings/hooks", () => ({
  useSettings: () => ({
    query_history_type: mockHistoryType,
    isLoading: false,
  }),
}));

function historyResponse(question = "Saved question", total = 200): Response {
  return {
    ok: true,
    status: 200,
    json: async () => ({
      items: total
        ? [
            {
              id: "93f34c4c-d49e-4f3a-86e0-2d989ac7fcd5",
              first_user_message: question,
              first_ai_message: "Saved answer",
              user_email: "person@example.com",
              assistant_name: "ASv3 Tek",
              model_display_names: [],
              time_created: "2026-10-20T12:00:00Z",
            },
          ]
        : [],
      total_items: total,
    }),
  } as Response;
}

beforeEach(() => {
  mockSearch = "";
  mockHistoryType = QueryHistoryType.NORMAL;
  mockReplace.mockClear();
  mockPush.mockClear();
});

afterEach(() => jest.restoreAllMocks());

test("restores exact user selection from the URL and applies searches across all pages", async () => {
  const user = setupUser();
  mockSearch =
    "user_email=person%40example.com&page=5&feedback_type=like&session_rating=4";
  const fetchSpy = jest
    .spyOn(global, "fetch")
    .mockResolvedValue(historyResponse());
  const view = render(<QueryHistoryTable />);
  await screen.findByText("Saved question");
  const initial = new URL(
    String(fetchSpy.mock.calls[0]![0]),
    "http://localhost"
  );
  expect(initial.searchParams.get("user_email")).toBe("person@example.com");
  expect(initial.searchParams.get("page_num")).toBe("4");
  expect(screen.getByLabelText("Kullanıcı e-postası / adı")).toHaveValue(
    "person@example.com"
  );
  await user.clear(screen.getByLabelText("Kullanıcı e-postası / adı"));
  await user.type(
    screen.getByLabelText("Kullanıcı e-postası / adı"),
    "Çağrı Şahin"
  );
  await user.type(screen.getByLabelText("Soru / sohbet ara"), "duty %_");
  expect(fetchSpy).toHaveBeenCalledTimes(1);
  await user.click(screen.getByRole("button", { name: "Ara" }));
  const destination = new URL(
    mockReplace.mock.calls.at(-1)![0],
    "http://localhost"
  );
  expect(destination.searchParams.get("page")).toBe("1");
  expect(destination.searchParams.get("user_query")).toBe("Çağrı Şahin");
  expect(destination.searchParams.has("user_email")).toBe(false);
  expect(destination.searchParams.get("q")).toBe("duty %_");
  expect(destination.searchParams.get("feedback_type")).toBe("like");
  expect(destination.searchParams.get("session_rating")).toBe("4");
  mockSearch = destination.search.slice(1);
  view.rerender(<QueryHistoryTable />);
  await waitFor(() => expect(fetchSpy).toHaveBeenCalledTimes(2));
  const filtered = new URL(
    String(fetchSpy.mock.calls.at(-1)![0]),
    "http://localhost"
  );
  expect(filtered.searchParams.get("user_query")).toBe("Çağrı Şahin");
  expect(filtered.searchParams.get("page_num")).toBe("0");
});

test("pagination preserves applied filters and clearing resets the full filter set", async () => {
  const user = setupUser();
  mockSearch =
    "user_query=Ayse&q=duty&api_key_query=Portal&start_time=2026-10-01T00%3A00%3A00Z&end_time=2026-10-23T00%3A00%3A00Z";
  jest.spyOn(global, "fetch").mockResolvedValue(historyResponse());
  render(<QueryHistoryTable />);
  await screen.findByText("Saved question");
  await user.click(screen.getByText("›"));
  const destination = new URL(
    mockReplace.mock.calls.at(-1)![0],
    "http://localhost"
  );
  expect(destination.searchParams.get("page")).toBe("2");
  expect(destination.searchParams.get("user_query")).toBe("Ayse");
  expect(destination.searchParams.get("q")).toBe("duty");
  expect(destination.searchParams.get("api_key_query")).toBe("Portal");
  expect(destination.searchParams.has("start_time")).toBe(true);
  await user.click(screen.getByRole("button", { name: "Filtreleri temizle" }));
  expect(mockReplace.mock.calls.at(-1)![0]).toBe(
    "/admin/performance/query-history?page=1"
  );
});

test("anonymized history hides identity controls and does not send identity URL filters", async () => {
  mockHistoryType = QueryHistoryType.ANONYMIZED;
  mockSearch =
    "user_email=private%40example.com&user_query=Private&api_key_query=Secret&q=duty";
  const fetchSpy = jest
    .spyOn(global, "fetch")
    .mockResolvedValue(historyResponse("", 0));
  render(<QueryHistoryTable />);
  await screen.findByText("Bu filtrelere uygun sohbet bulunamadı.");
  expect(
    screen.queryByLabelText("Kullanıcı e-postası / adı")
  ).not.toBeInTheDocument();
  expect(screen.queryByLabelText("API anahtarı")).not.toBeInTheDocument();
  const request = new URL(
    String(fetchSpy.mock.calls[0]![0]),
    "http://localhost"
  );
  expect(request.searchParams.has("user_email")).toBe(false);
  expect(request.searchParams.has("user_query")).toBe(false);
  expect(request.searchParams.has("api_key_query")).toBe(false);
  expect(request.searchParams.get("q")).toBe("duty");
  expect(screen.getByText("0 sohbet bulundu")).toBeInTheDocument();
});

test("late responses from a previous filter cannot replace the active search", async () => {
  const oldResponse = deferred<Response>();
  const fetchSpy = jest
    .spyOn(global, "fetch")
    .mockImplementation(async (input) => {
      const url = new URL(String(input), "http://localhost");
      return url.searchParams.get("q") === "old"
        ? oldResponse.promise
        : historyResponse("Current filtered question");
    });
  mockSearch = "q=old";
  const view = render(<QueryHistoryTable />);
  await waitFor(() => expect(fetchSpy).toHaveBeenCalledTimes(1));
  mockSearch = "q=new";
  view.rerender(<QueryHistoryTable />);
  await screen.findByText("Current filtered question");
  await act(async () => oldResponse.resolve(historyResponse("Old question")));
  expect(screen.queryByText("Old question")).not.toBeInTheDocument();
  expect(screen.getByText("Current filtered question")).toBeInTheDocument();
});
