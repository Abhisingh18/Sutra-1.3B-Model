"use client";

import { useEffect, useRef, useState } from "react";

// Set in Vercel: Settings -> Environment Variables. This is the cloudflared
// URL printed by deploy/server.py's tunnel, and it changes on every restart
// unless you use a named tunnel.
const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Src = { score: number; text: string; name: string };
type Msg = { role: "user" | "assistant"; text: string; sources?: Src[] };

// Chosen by testing, not by guessing. Every candidate was run through the
// model and only the ones that produced usable output survived: lists and
// short notes work, while "summarise this" invented dates that were not in
// the text and "rewrite this politely" answered with a riddle. Trivia is
// absent on purpose -- 18B training tokens do not buy reliable facts.
const EXAMPLES = [
  {
    title: "Write a note",
    body: "Write a thank you note to a colleague who helped me finish a project.",
  },
  {
    title: "Give me tips",
    body: "List five tips for studying effectively.",
  },
  {
    title: "Make bullet points",
    body: "Write five short bullet points about healthy eating.",
  },
  {
    title: "Draft an email",
    body: "Write a short email to my manager asking for two days of leave.",
  },
];

export default function Home() {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [online, setOnline] = useState<boolean | null>(null);
  // No Wikipedia retrieval. Four indexes were built -- 60k, 500k, 400k
  // shuffled, 150k length-filtered -- and recall@3 never moved off 50%.
  // The cause is arithmetic, not a bug: any few-hundred-thousand sample is
  // 2-6% of Wikipedia, so a question about one specific article misses most
  // of the time, and a confident wrong passage is worse than none. Covering
  // it properly means ~29 GB of embeddings and a disk-backed index.
  //
  // Uploaded documents have no such problem: coverage is total by
  // construction, which is what makes that the feature worth shipping.
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
              It writes and rewrites well; for anything factual, upload a
              document and it will answer from that.
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
          {!doc && (
            <span className="hint">
              Upload a document to get answers grounded in it
            </span>
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
