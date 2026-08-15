"use client";

import { ReactNode } from "react";

/* A deliberately small Markdown renderer.
 *
 * The model writes numbered lists, bullets and the occasional bold run, and
 * nothing else -- it has never produced a table or a nested list in testing.
 * Pulling in a full parser would add far more bundle weight than those three
 * constructs justify, and every dependency is one more thing that can break a
 * build on a page whose whole job is to render streaming text.
 *
 * Written to be safe by construction: text goes through React as children, so
 * it is escaped, and no branch ever builds HTML from a string.
 */

function inline(text: string, key: string): ReactNode[] {
  const out: ReactNode[] = [];
  // Bold, italics and inline code in one pass, so a run like **`x`** does not
  // depend on which pattern is applied first.
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;

  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const tok = m[0];
    const k = `${key}-i${i++}`;
    if (tok.startsWith("**")) out.push(<strong key={k}>{tok.slice(2, -2)}</strong>);
    else if (tok.startsWith("`")) out.push(<code key={k}>{tok.slice(1, -1)}</code>);
    else out.push(<em key={k}>{tok.slice(1, -1)}</em>);
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function Markdown({ text }: { text: string }) {
  const lines = text.split("\n");
  const blocks: ReactNode[] = [];

  // Buffers for the block currently being accumulated. Lists have to be
  // gathered across lines rather than emitted per line, or each item becomes
  // its own single-item list.
  let para: string[] = [];
  let items: string[] = [];
  let ordered = false;
  let code: string[] | null = null;
  let n = 0;

  const flushPara = () => {
    if (!para.length) return;
    const t = para.join(" ");
    blocks.push(<p key={`p${n++}`}>{inline(t, `p${n}`)}</p>);
    para = [];
  };

  const flushList = () => {
    if (!items.length) return;
    const kids = items.map((it, j) => <li key={j}>{inline(it, `l${n}-${j}`)}</li>);
    blocks.push(
      ordered ? <ol key={`l${n++}`}>{kids}</ol> : <ul key={`l${n++}`}>{kids}</ul>
    );
    items = [];
  };

  for (const raw of lines) {
    const line = raw.trimEnd();

    if (line.trim().startsWith("```")) {
      if (code === null) {
        flushPara();
        flushList();
        code = [];
      } else {
        blocks.push(
          <pre key={`c${n++}`}>
            <code>{code.join("\n")}</code>
          </pre>
        );
        code = null;
      }
      continue;
    }
    if (code !== null) {
      code.push(raw);
      continue;
    }

    const num = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
    const bul = line.match(/^\s*[-*•]\s+(.*)$/);
    const head = line.match(/^\s*(#{1,3})\s+(.*)$/);

    if (num || bul) {
      flushPara();
      const isOrdered = Boolean(num);
      // A bullet list directly after a numbered one is a different block.
      if (items.length && isOrdered !== ordered) flushList();
      ordered = isOrdered;
      items.push((num ? num[2] : bul![1]).trim());
      continue;
    }

    flushList();

    if (head) {
      flushPara();
      const H = (head[1].length === 1 ? "h3" : "h4") as "h3" | "h4";
      blocks.push(<H key={`h${n++}`}>{inline(head[2], `h${n}`)}</H>);
      continue;
    }

    if (!line.trim()) flushPara();
    else para.push(line.trim());
  }

  if (code !== null && code.length)
    blocks.push(
      <pre key={`c${n++}`}>
        <code>{code.join("\n")}</code>
      </pre>
    );
  flushList();
  flushPara();

  return <div className="md">{blocks}</div>;
}
