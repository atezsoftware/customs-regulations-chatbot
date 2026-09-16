"use client";

import { useEffect, useState, type ReactNode } from "react";
import { CompactMarkdown, Text } from "@opal/components";
import type { RichStr } from "@opal/types";

interface ChunkContentProps {
  text: string;
}

/** Render only table structure and text. Source HTML never reaches innerHTML. */
function tableNode(node: Node, key: string): ReactNode {
  if (node.nodeType === Node.TEXT_NODE) return node.textContent;
  if (!(node instanceof Element)) return null;
  const tag = node.tagName.toLowerCase();
  if (["script", "style", "iframe", "object", "img", "svg"].includes(tag))
    return null;
  const children = Array.from(node.childNodes).map((child, index) =>
    tableNode(child, `${key}-${index}`)
  );
  const span = (name: string) =>
    Math.min(1000, Math.max(1, Number(node.getAttribute(name)) || 1));
  switch (tag) {
    case "table":
      return (
        <table key={key} className="w-full border-collapse">
          {children}
        </table>
      );
    case "thead":
      return <thead key={key}>{children}</thead>;
    case "tbody":
      return <tbody key={key}>{children}</tbody>;
    case "tfoot":
      return <tfoot key={key}>{children}</tfoot>;
    case "tr":
      return <tr key={key}>{children}</tr>;
    case "th":
      return (
        <th
          key={key}
          rowSpan={span("rowspan")}
          colSpan={span("colspan")}
          className="border border-border-02 p-2 text-left"
        >
          {children}
        </th>
      );
    case "td":
      return (
        <td
          key={key}
          rowSpan={span("rowspan")}
          colSpan={span("colspan")}
          className="border border-border-02 p-2"
        >
          {children}
        </td>
      );
    case "br":
      return <br key={key} />;
    default:
      return <span key={key}>{children}</span>;
  }
}

export function ChunkContent({ text }: ChunkContentProps) {
  const [table, setTable] = useState<ReactNode>(null);
  useEffect(() => {
    setTable(
      /<table[\s>]/i.test(text)
        ? tableNode(
            new DOMParser().parseFromString(text, "text/html").body,
            "source"
          )
        : null
    );
  }, [text]);
  return table ? (
    <div className="overflow-x-auto text-text-04">{table}</div>
  ) : (
    <CompactMarkdown>{text}</CompactMarkdown>
  );
}

function ChangedText({
  text,
  other,
  side,
}: {
  text: string;
  other: string;
  side: "Before" | "After";
}) {
  if (text === other || /<table[\s>]|^\s*[#|>*-]/im.test(text))
    return <ChunkContent text={text} />;
  const current = Array.from(text),
    previous = Array.from(other);
  let start = 0,
    end = 0;
  while (
    start < Math.min(current.length, previous.length) &&
    current[start] === previous[start]
  )
    start++;
  while (
    end < Math.min(current.length, previous.length) - start &&
    current[current.length - 1 - end] === previous[previous.length - 1 - end]
  )
    end++;
  return (
    <div className="whitespace-pre-wrap text-text-04">
      {current.slice(0, start).join("")}
      <mark
        className={
          side === "Before"
            ? "bg-status-error-01 text-text-04"
            : "bg-status-success-01 text-text-04"
        }
      >
        {current.slice(start, current.length - end).join("")}
      </mark>
      {end > 0 ? current.slice(-end).join("") : ""}
    </div>
  );
}

interface ChunkChangeCardProps {
  title: string | RichStr;
  before: string[];
  after: string[];
  beforeImages?: string[];
  afterImages?: string[];
  actions?: ReactNode;
  children?: ReactNode;
}

export default function ChunkChangeCard({
  title,
  before,
  after,
  beforeImages = [],
  afterImages = [],
  actions,
  children,
}: ChunkChangeCardProps) {
  return (
    <section className="flex flex-col gap-3 rounded-12 border border-border-02 bg-background-neutral-00 p-4">
      <Text as="h3" font="main-ui-action">
        {title}
      </Text>
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        {[
          { title: "Before" as const, chunks: before, images: beforeImages },
          { title: "After" as const, chunks: after, images: afterImages },
        ].map((side) => (
          <div key={side.title} className="min-w-0 space-y-2">
            <Text font="secondary-action">{side.title}</Text>
            {side.chunks.length === 0 && (
              <Text font="secondary-body">
                {side.title === "Before" ? "New chunk" : "Removed"}
              </Text>
            )}
            {side.chunks.map((text, index) => (
              <ChangedText
                key={index}
                text={text}
                other={(side.title === "Before" ? after : before)[index] ?? ""}
                side={side.title}
              />
            ))}
            {side.images.map((url) => (
              <a key={url} href={url} target="_blank" rel="noreferrer">
                <img
                  src={url}
                  alt={`${side.title} source evidence`}
                  loading="lazy"
                  className="max-h-80 max-w-full object-contain"
                />
              </a>
            ))}
          </div>
        ))}
      </div>
      {children}
      <div className="flex flex-wrap gap-2">{actions}</div>
    </section>
  );
}
