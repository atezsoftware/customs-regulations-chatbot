import { render, screen } from "@tests/setup/test-utils";
import QueryHistoryUser from "@/app/ee/admin/performance/query-history/QueryHistoryUser";

test("shows the supplied Turkish name and the originating API key", () => {
  render(
    <QueryHistoryUser
      session={{
        user_name: "Çağrı Şahin Özışık",
        user_email: "api-key@example.com",
        api_key_id: 12,
        api_key_name: "Müşteri uygulaması",
        api_key_display: "dn_****1234",
      }}
    />
  );

  expect(screen.getByText("Çağrı Şahin Özışık")).toBeInTheDocument();
  expect(
    screen.getByText("Müşteri uygulaması · dn_****1234")
  ).toBeInTheDocument();
  expect(screen.queryByText("api-key@example.com")).not.toBeInTheDocument();
});

test("keeps the email fallback when a name was not supplied", () => {
  render(
    <QueryHistoryUser
      session={{
        user_email: "api-key@example.com",
        api_key_id: 12,
        api_key_display: "dn_****1234",
      }}
    />
  );

  expect(screen.getByText("api-key@example.com")).toBeInTheDocument();
  expect(screen.getByText("API Key #12 · dn_****1234")).toBeInTheDocument();
});

test("keeps ordinary user sessions free of API key metadata", () => {
  render(<QueryHistoryUser session={{ user_email: "user@example.com" }} />);

  expect(screen.getByText("user@example.com")).toBeInTheDocument();
  expect(screen.queryByText(/API Key/)).not.toBeInTheDocument();
});

test("renders a supplied name as literal text", () => {
  render(
    <QueryHistoryUser
      session={{
        user_name: "**Çağrı Şahin** <script>alert(1)</script>",
        user_email: null,
      }}
    />
  );

  expect(
    screen.getByText("**Çağrı Şahin** <script>alert(1)</script>")
  ).toBeInTheDocument();
  expect(screen.queryByRole("link")).not.toBeInTheDocument();
});
