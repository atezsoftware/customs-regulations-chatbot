import { render, screen, setupUser, waitFor } from "@tests/setup/test-utils";
import QueryHistoryApiKeyFilter from "@/app/ee/admin/performance/query-history/QueryHistoryApiKeyFilter";

beforeAll(() => {
  Element.prototype.scrollIntoView = jest.fn();
});

beforeEach(() =>
  jest.spyOn(global, "fetch").mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => [
      { api_key_id: 17, api_key_name: "Tariff", api_key_display: "masked" },
      { api_key_id: 18, api_key_name: "Other API", api_key_display: "masked" },
    ],
  } as Response)
);
afterEach(() => jest.restoreAllMocks());

test("defaults to all APIs and applies any selected API immediately, preserving user and dates", async () => {
  const user = setupUser();
  const onApply = jest.fn();
  const filters = {
    user_query: "Ayşe",
    start_time: "2026-11-01T00:00:00Z",
    end_time: "2026-11-09T00:00:00Z",
  };
  const view = render(
    <QueryHistoryApiKeyFilter filters={filters} onApply={onApply} />
  );
  const picker = await screen.findByRole("combobox", {
    name: "API / İstek kaynağı",
  });
  await waitFor(() => expect(picker).toBeEnabled());
  expect(picker).toHaveTextContent("Tüm API’ler");
  expect(onApply).not.toHaveBeenCalled();
  picker.focus();
  await user.keyboard("{Enter}");
  (await screen.findByRole("option", { name: /Other API · #18/ })).focus();
  await user.keyboard("{Enter}");
  expect(onApply).toHaveBeenLastCalledWith({ ...filters, api_key_id: 18 });
  view.rerender(
    <QueryHistoryApiKeyFilter
      filters={{ ...filters, api_key_id: 18 }}
      onApply={onApply}
    />
  );
  screen.getByRole("combobox", { name: "API / İstek kaynağı" }).focus();
  await user.keyboard("{Enter}");
  (await screen.findByRole("option", { name: /Tüm API’ler/ })).focus();
  await user.keyboard("{Enter}");
  expect(onApply).toHaveBeenLastCalledWith(filters);
});
