import { selectionCitationClipboard } from "@/app/app/message/copyingUtils";

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
  expect(copied?.html).toContain('<a href="/api/asv3/citation/2/1">[1]</a>');
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
