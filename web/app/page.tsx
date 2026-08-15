"use client";

import { useEffect, useRef, useState } from "react";

// Set in Vercel: Settings -> Environment Variables. This is the cloudflared
// URL printed by `deploy/server.py`'s tunnel, and it changes each time the
// tunnel restarts unless you use a named tunnel.
const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Msg = { role: "user" | "assistant"; text: string };

const EXAMPLES = [
  "Write a short email to my manager asking for two days of leave.",
  "Explain photosynthesis in three sentences.",
  "What is machine learning?",
  "List five healthy breakfast foods.",
];

export default function Home() {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [online, setOnline] = useState<boolean | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  // The backend runs on someone's workstation behind a tunnel, so "is it up"
  // is a real question the UI has to answer -- otherwise a dead tunnel looks
  // identical to a slow model.
  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => setOnline(r.ok))
      .catch(() => setOnline(false));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs]);

  async function send(text: string) {
    if (!text.trim() || busy) return;
    setInput("");
    setMsgs((m) => [...m, { role: "user", text }, { role: "assistant", text: "" }]);
    setBusy(true);

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, max_tokens: 160, temperature: 0.5 }),
      });
      if (!res.body) throw new Error("no stream");

      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        // SSE frames are separated by a blank line; a chunk can split one in
        // half, so keep the remainder in the buffer.
        const frames = buf.split("\n\n");
        buf = frames.pop() || "";
        for (const f of frames) {
          if (!f.startsWith("data: ")) continue;
          const d = JSON.parse(f.slice(6));
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
          text: "Could not reach the model server. It runs on a workstation behind a tunnel and may be offline.",
        };
        return c;
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <header>
        <h1>Sutra-1.3B</h1>
        <p className="sub">
          A 1.32B-parameter Mixture-of-Experts model trained from scratch on 18B
          tokens. 0.28B parameters active per token.
        </p>
        <div className="links">
          <a href="https://github.com/Abhisingh18/Sutra-1.3B-Model">GitHub</a>
          <a href="https://huggingface.co/Abhisingh-18/Sutra-1.3B-Chat">Weights</a>
          <span className={`dot ${online === null ? "" : online ? "up" : "down"}`} />
          <span className="status">
            {online === null ? "checking" : online ? "online" : "offline"}
          </span>
        </div>
      </header>

      <div className="chat">
        {msgs.length === 0 && (
          <div className="empty">
            <p>Try one of these — it responds best to full, explicit sentences.</p>
            {EXAMPLES.map((e) => (
              <button key={e} onClick={() => send(e)} className="example">
                {e}
              </button>
            ))}
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="bubble">
              {m.text || <span className="cursor">▍</span>}
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask something…"
          disabled={busy}
        />
        <button type="submit" disabled={busy || !input.trim()}>
          {busy ? "…" : "Send"}
        </button>
      </form>

      <footer>
        Trained on 18B tokens — roughly 500x less than comparable 1B models. It
        writes fluently but does not reliably know facts.
      </footer>
    </main>
  );
}
