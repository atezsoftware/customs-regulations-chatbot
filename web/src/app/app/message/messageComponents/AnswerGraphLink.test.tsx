import { fireEvent, render, screen } from "@testing-library/react";
import AnswerGraphLink from "./AnswerGraphLink";

let mockIsAdmin = false;

jest.mock("@/providers/UserProvider", () => ({
  useUser: () => ({ isAdmin: mockIsAdmin }),
}));

jest.mock("@opal/components", () => ({
  Button: ({
    children,
    onClick,
  }: React.PropsWithChildren<{
    onClick: React.MouseEventHandler<HTMLButtonElement>;
  }>) => <button onClick={onClick}>{children}</button>,
}));

describe("AnswerGraphLink", () => {
  afterEach(() => {
    mockIsAdmin = false;
    jest.restoreAllMocks();
  });

  it("does not offer graph access to a non-admin", () => {
    render(<AnswerGraphLink messageId={42} />);
    expect(screen.queryByText("Execution graph")).not.toBeInTheDocument();
  });

  it("opens the selected assistant message graph in a new tab", () => {
    mockIsAdmin = true;
    const open = jest.spyOn(window, "open").mockImplementation(() => null);
    render(<AnswerGraphLink messageId={42} />);
    fireEvent.click(screen.getByRole("button", { name: "Execution graph" }));
    expect(open).toHaveBeenCalledWith(
      "/admin/answer-graphs/message/42",
      "_blank",
      "noopener,noreferrer"
    );
  });
});
