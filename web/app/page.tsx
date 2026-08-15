"use client";

import { useEffect, useRef, useState } from "react";

// Set in Vercel: Settings -> Environment Variables. This is the cloudflared
// URL printed by deploy/server.py's tunnel, and it changes on every restart
// unless you use a named tunnel.
const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Src = { score: number; text: string; name: string };
type Msg = { role: "user" | "assistant"; text: string; sources?: Src[] };

const EXAMPLES = [
  {
    title: "Write an email",
    body: "Write a short email to my manager asking for two days of leave.",
  },
  {
    title: "Explain a concept",
    body: "Explain photosynthesis in three sentences.",
  },
  {
    title: "Summarise a topic",
    body: "What is machine learning?",
  },
  {
    title: "Make a list",
    body: "List five healthy breakfast foods.",
  },
];

export default function Home() {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [online, setOnline] = useState<boolean | null>(null);
  // Off by default until retrieval coverage improves. Measured recall@3 on the
  // current index is 50%, and a miss is worse than no context at all: asked
  // "what is light" it retrieved a plant passage and answered about plants,
  // where with no context it at least stayed on topic.
  const [rag, setRag] = useState(false);
  const [hasRag, setHasRag] = useState(false);
  const [canUpload, setCanUpload] = useState(false);
  const [doc, setDoc] = useState<{ id: string; name: string; chunks: number } | null>(null);
  const [uploading, setUploading] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  // The backend runs on a workstation behind a tunnel, so "is it up" is a real
  // question the UI must answer -- a dead tunnel otherwise looks identical to a
  // slow model.
  useEffect(() => {
    const ping = () =>
      fetch(`${API}/health`)
        .then(async (r) => {
          setOnline(r.ok);
          if (r.ok) {
            const h = await r.json();
            setHasRag(Boolean(h.rag));
            setCanUpload(Boolean(h.upload));
          }
        })
        .catch(() => setOnline(false));
    ping();
    const t = setInterval(ping, 30000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs]);

  function grow() {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  }

  async function send(text: string) {
    if (!text.trim() || busy) return;
    setInput("");
    if (taRef.current) taRef.current.style.height = "auto";
    setMsgs((m) => [...m, { role: "user", text }, { role: "assistant", text: "" }]);
    setBusy(true);

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          max_tokens: 512,
          temperature: 0.5,
          rag: rag && hasRag,
          doc_id: doc?.id ?? null,
        }),
      });
      if (!res.body) throw new Error("no stream");

      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        // SSE frames are separated by a blank line, and a network chunk can
        // split one in half -- keep the remainder for the next round.
        const frames = buf.split("\n\n");
        buf = frames.pop() || "";
        for (const f of frames) {
          if (!f.startsWith("data: ")) continue;
          const d = JSON.parse(f.slice(6));
          if (d.sources?.length) {
            setMsgs((m) => {
              const c = [...m];
              c[c.length - 1] = { ...c[c.length - 1], sources: d.sources };
              return c;
            });
          }
          if (d.token) {
            setMsgs((m) => {
              const c = [...m];
              c[c.length - 1] = {
                role: "assistant",
                text: c[c.length - 1].text + d.token,
              };
              return c;
            });
          }
        }
      }
    } catch {
      setMsgs((m) => {
        const c = [...m];
        c[c.length - 1] = {
          role: "assistant",
          text: "Could not reach the model server. It runs on a workstation behind a tunnel and may be offline right now.",
        };
        return c;
      });
    } finally {
      setBusy(false);
    }
  }

  async function upload(f: File) {
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", f);
      const r = await fetch(`${API}/upload`, { method: "POST", body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || "upload failed");
      setDoc({ id: d.doc_id, name: d.name, chunks: d.chunks });
      setMsgs((m) => [
        ...m,
        {
          role: "assistant",
          text: `Indexed ${d.name} — ${d.chunks} passages. Ask me about it; I will answer from the document rather than from memory.`,
        },
      ]);
    } catch (e) {
      setMsgs((m) => [
        ...m,
        { role: "assistant", text: `Could not read that file. ${(e as Error).message}` },
      ]);
    } finally {
      setUploading(false);
    }
  }

  const empty = msgs.length === 0;

  return (
    <div className="shell">
      <nav>
        <div className="brand">
          <span className="mark">स</span>
          <span className="name">Sutra</span>
          <span className="badge">1.3B</span>
        </div>
        <div className="navlinks">
          <span className={`status ${online ? "up" : online === false ? "down" : ""}`}>
            <i /> {online === null ? "checking" : online ? "online" : "offline"}
          </span>
          <a href="https://github.com/Abhisingh18/Sutra-1.3B-Model">GitHub</a>
          <a href="https://huggingface.co/Abhisingh-18/Sutra-1.3B-Chat">Weights</a>
        </div>
      </nav>

      <main className={empty ? "centered" : ""}>
        {empty ? (
          <div className="hero">
            <h1>
              Trained from scratch.
              <br />
              <span className="dim">Ask it anything.</span>
            </h1>
            <p className="lede">
              A 1.32B-parameter Mixture-of-Experts model pretrained on 18B tokens,
              then tuned with SFT and DPO. Only 0.28B parameters run per token.
            </p>
            <div className="cards">
              {EXAMPLES.map((e) => (
                <button key={e.body} className="card" onClick={() => send(e.body)}>
                  <strong>{e.title}</strong>
                  <span>{e.body}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="thread">
            {msgs.map((m, i) => (
              <div key={i} className={`turn ${m.role}`}>
                <div className="who">{m.role === "user" ? "You" : "Sutra"}</div>
                <div className="body">
                  {m.text || <span className="dots"><i /><i /><i /></span>}
                </div>
                {m.sources?.length ? (
                  <details className="sources">
                    <summary>
                      Answered from {m.sources.length} passage
                      {m.sources.length > 1 ? "s" : ""} — check it
                    </summary>
                    {m.sources.map((s, j) => (
                      <blockquote key={j}>
                        <cite>
                          {s.name} · {s.score}
                        </cite>
                        {s.text}
                      </blockquote>
                    ))}
                  </details>
                ) : null}
              </div>
            ))}
            <div ref={endRef} />
          </div>
        )}
      </main>

      <div className="composer-wrap">
        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <textarea
            ref={taRef}
            value={input}
            rows={1}
            placeholder="Ask Sutra anything…"
            onChange={(e) => {
              setInput(e.target.value);
              grow();
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(input);
              }
            }}
          />
          <button type="submit" disabled={busy || !input.trim()} aria-label="Send">
            {busy ? <span className="spin" /> : "↑"}
          </button>
        </form>
        <div className="tools">
          {canUpload && (
            <label className="tool">
              <input
                type="file"
                accept=".txt,.md,.pdf"
                hidden
                disabled={uploading}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) upload(f);
                  e.target.value = "";
                }}
              />
              {uploading ? "Indexing…" : "＋ Upload a document"}
            </label>
          )}
          {doc && (
            <span className="chip">
              {doc.name} · {doc.chunks} passages
              <button onClick={() => setDoc(null)} aria-label="Remove">×</button>
            </span>
          )}
          {hasRag && !doc && (
            <button
              className={`tool ${rag ? "on" : ""}`}
              onClick={() => setRag(!rag)}
              title="Look the answer up in Wikipedia before replying"
            >
              {rag ? "✓ " : ""}Wikipedia
            </button>
          )}
        </div>
        <p className="disclaimer">
          Trained on 18B tokens — about 500x less than comparable 1B models. It
          writes fluently but does not reliably know facts.
        </p>
      </div>
    </div>
  );
}
