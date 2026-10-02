"use client";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkRehype from "remark-rehype";
import rehypeHighlight from "rehype-highlight";
import { loadHighlightLanguages } from "@/lib/highlightLanguages";
import rehypeKatex from "rehype-katex";
import rehypeSanitize from "rehype-sanitize";
import rehypeStringify from "rehype-stringify";

export function selectionCitationClipboard(
  fragment: DocumentFragment
): { html: string; text: string } | null {
  const citations = fragment.querySelectorAll<HTMLElement>(
    "[data-citation-copy-text]"
  );
  if (!citations.length) return null;
  for (const citation of Array.from(citations)) {
    const label = citation.dataset.citationCopyText ?? "";
    if (!/^\[(?:D|Q)?\d+\]$/.test(label)) continue;
    const target = citation.dataset.citationCopyHref ?? "";
    const replacement = document.createElement("a");
    replacement.textContent = label;
    if (/^https?:\/\//i.test(target) || /^\/(?!\/)/.test(target)) {
      replacement.setAttribute("href", target);
    }
    citation.replaceWith(replacement);
  }
  const container = document.createElement("div");
  container.appendChild(fragment);
  function plainText(node: Node): string {
    if (node.nodeType === Node.TEXT_NODE) return node.textContent ?? "";
    if (node instanceof HTMLElement && node.tagName === "BR") return "\n";
    const text = Array.from(node.childNodes).map(plainText).join("");
    return node instanceof HTMLElement &&
      /^(P|DIV|LI|H[1-6]|TR|PRE)$/.test(node.tagName)
      ? text + "\n"
      : text;
  }
  return { html: container.innerHTML, text: plainText(container).trimEnd() };
}

export function handleCopy(
  event: Pick<
    ClipboardEvent,
    "clipboardData" | "preventDefault" | "defaultPrevented"
  >,
  markdownRef: React.RefObject<HTMLDivElement | null>
) {
  if (event.defaultPrevented || !event.clipboardData) return;
  // Check if we have a selection
  const selection = window.getSelection();
  if (!selection?.rangeCount) return;

  const range = selection.getRangeAt(0);

  // If selection is within our markdown container
  if (
    markdownRef.current &&
    markdownRef.current.contains(range.commonAncestorContainer)
  ) {
    event.preventDefault();

    // Clone selection to get the HTML
    const fragment = range.cloneContents();
    const citationContent = selectionCitationClipboard(fragment);
    if (citationContent) {
      event.clipboardData.setData("text/html", citationContent.html);
      event.clipboardData.setData("text/plain", citationContent.text);
    } else {
      const tempDiv = document.createElement("div");
      tempDiv.appendChild(fragment);
      event.clipboardData.setData("text/html", tempDiv.innerHTML);
      event.clipboardData.setData("text/plain", selection.toString());
    }
  }
}

export function registerMarkdownCopyHandler(
  markdownRef: React.RefObject<HTMLDivElement | null>
): () => void {
  // Native copy may target the focused input/body rather than the selected prose.
  const listener = (event: ClipboardEvent) => handleCopy(event, markdownRef);
  document.addEventListener("copy", listener);
  return () => document.removeEventListener("copy", listener);
}

export function citationClipboardHtml(html: string): string {
  const template = document.createElement("template");
  template.innerHTML = html;
  return selectionCitationClipboard(template.content)?.html ?? html;
}

// Convert markdown tables to TSV format for spreadsheet compatibility
export function convertMarkdownTablesToTsv(content: string): string {
  const lines = content.split("\n");
  const result: string[] = [];

  for (const line of lines) {
    // Check if line is a markdown table row (starts and ends with |)
    const trimmed = line.trim();
    if (trimmed.startsWith("|") && trimmed.endsWith("|")) {
      // Check if it's a divider row (contains only |, -, :, and spaces)
      if (/^\|[\s\-:|\s]+\|$/.test(trimmed)) {
        // Skip divider rows
        continue;
      }
      // Convert table row: split by |, trim cells, join with tabs
      const placeholder = "\x00";
      const cells = trimmed
        .slice(1, -1) // Remove leading and trailing |
        .replace(/\\\|/g, placeholder) // Preserve escaped pipes
        .split("|")
        .map((cell) => cell.trim().replace(new RegExp(placeholder, "g"), "|"));
      result.push(cells.join("\t"));
    } else {
      result.push(line);
    }
  }

  return result.join("\n");
}

// For copying the entire content
export function copyAll(content: string) {
  // Convert markdown to HTML using unified ecosystem. Grammars load dynamically
  // so the highlight.js corpus stays out of the chat bundle.
  loadHighlightLanguages().then((languages) => {
    unified()
      .use(remarkParse)
      .use(remarkGfm)
      .use(remarkMath)
      .use(remarkRehype)
      .use(rehypeHighlight, { languages })
      .use(rehypeKatex)
      .use(rehypeSanitize)
      .use(rehypeStringify)
      .process(content)
      .then((file: any) => {
        const htmlContent = String(file);

        // Create clipboard data
        const clipboardItem = new ClipboardItem({
          "text/html": new Blob([htmlContent], { type: "text/html" }),
          "text/plain": new Blob([content], { type: "text/plain" }),
        });

        navigator.clipboard.write([clipboardItem]);
      });
  });
}
