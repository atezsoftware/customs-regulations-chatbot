import {
  selectionCitationClipboard,
  registerMarkdownCopyHandler,
  citationClipboardHtml,
} from "@/app/app/message/copyingUtils";

function fragment(html: string): DocumentFragment {
  const template = document.createElement("template");
  template.innerHTML = html;
  return template.content;
}

it("copies inline source numbers and links without chip folder labels", () => {
  const copied = selectionCitationClipboard(
    fragment(
      '<p>First rule <span data-citation-copy-text="[1]" data-citation-copy-href="/api/asv3/citation/2/1"><button>GLOBAL Long/folder/name</button></span>.</p><p>Later settlement <span data-citation-copy-text="[2]" data-citation-copy-href="https://example.test/law"><button>Another source</button></span>.</p>'
    )
  );
  expect(copied?.text).toBe("First rule [1].\nLater settlement [2].");
  expect(copied?.html).toContain(
    '<a href="http://localhost/api/asv3/citation/2/1">[1]</a>'
  );
  expect(copied?.html).not.toContain("GLOBAL");
  expect(copied?.html).not.toContain("Long/folder");
});

it("preserves legal text and does not rewrite ordinary selections", () => {
  expect(
    selectionCitationClipboard(
      fragment("<p>GLOBAL is a literal part of this text.</p>")
    )
  ).toBeNull();
  const copied = selectionCitationClipboard(
    fragment(
      '<p>GLOBAL term <span data-citation-copy-text="[3]"><button>Source label</button></span>.</p>'
    )
  );
  expect(copied?.text).toBe("GLOBAL term [3].");
});

it("does not turn unsafe citation destinations into clipboard links", () => {
  const copied = selectionCitationClipboard(
    fragment(
      '<span data-citation-copy-text="[4]" data-citation-copy-href="javascript:alert(1)">Source</span>'
    )
  );
  expect(copied?.html).toBe("<a>[4]</a>");
});

it("copies selected prose with source numbers even when the native event targets the body", () => {
  const answer = document.createElement("div");
  answer.innerHTML =
    '<p>Operative rule <span data-citation-copy-text="[7]" data-citation-copy-href="/api/asv3/citation/2/7"><button>GLOBAL Long/folder</button></span>.</p>';
  document.body.appendChild(answer);
  const range = document.createRange();
  range.selectNodeContents(answer);
  const selection = window.getSelection();
  selection?.removeAllRanges();
  selection?.addRange(range);
  const copied = new Map<string, string>();
  const event = new Event("copy", { bubbles: true, cancelable: true });
  Object.defineProperty(event, "clipboardData", {
    value: {
      setData: (type: string, value: string) => copied.set(type, value),
    },
  });
  const unregister = registerMarkdownCopyHandler({ current: answer });
  try {
    document.body.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);
    expect(copied.get("text/plain")).toBe("Operative rule [7].");
    expect(copied.get("text/html")).toContain(
      '<a href="http://localhost/api/asv3/citation/2/7">[7]</a>'
    );
    expect(copied.get("text/html")).not.toContain("GLOBAL");
  } finally {
    unregister();
    selection?.removeAllRanges();
    answer.remove();
  }
});

it("does not intercept outside selections or keep a detached message listener", () => {
  const answer = document.createElement("div");
  const outside = document.createElement("p");
  outside.textContent = "Another user selection";
  document.body.append(answer, outside);
  const selection = window.getSelection();
  const range = document.createRange();
  range.selectNodeContents(outside);
  selection?.removeAllRanges();
  selection?.addRange(range);
  const setData = jest.fn();
  const event = new Event("copy", { bubbles: true, cancelable: true });
  Object.defineProperty(event, "clipboardData", { value: { setData } });
  const unregister = registerMarkdownCopyHandler({ current: answer });
  try {
    document.body.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
    expect(setData).not.toHaveBeenCalled();
    unregister();
    range.selectNodeContents(answer);
    selection?.removeAllRanges();
    selection?.addRange(range);
    document.body.dispatchEvent(event);
    expect(setData).not.toHaveBeenCalled();
  } finally {
    unregister();
    selection?.removeAllRanges();
    answer.remove();
    outside.remove();
  }
});

it("uses numeric citations in full-answer HTML while preserving ordinary body words", () => {
  const html =
    '<p>GLOBAL is ordinary prose <span data-citation-copy-text="[8]"><button>Long/source/path</button></span>.</p>';
  expect(citationClipboardHtml(html)).toBe(
    "<p>GLOBAL is ordinary prose <a>[8]</a>.</p>"
  );
  expect(citationClipboardHtml("<p>No citations.</p>")).toBe(
    "<p>No citations.</p>"
  );
});
