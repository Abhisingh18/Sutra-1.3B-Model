"use client";

import { useEffect, useRef, useState } from "react";

import { Markdown } from "../markdown";
import { SidebarAccount } from "../authbutton";
import { useProfile } from "../profile";

import { backendUrl, forgetBackend } from "../backend";
const STORE_BASE = "sutra.chats.v1";

type Src = { score: number; text: string; name: string };
type Msg = { role: "user" | "assistant"; text: string; sources?: Src[] };
type Chat = { id: string; title: string; at: number; msgs: Msg[] };

// Chosen by testing, not by guessing. Every candidate was run through the
// model and only the ones that produced usable output survived: lists and
// short notes work, while "summarise this" invented dates that were not in
// the text and "rewrite this politely" answered with a riddle. Trivia is
// absent on purpose -- 18B training tokens do not buy reliable facts.
const EXAMPLES = [
  { title: "Write a note", body: "Write a thank you note to a colleague who helped me finish a project." },
  { title: "Give me tips", body: "List five tips for studying effectively." },
  { title: "Make bullet points", body: "Write five short bullet points about healthy eating." },
  { title: "Draft an email", body: "Write a short email to my manager asking for two days of leave." },
];

const newId = () => Math.random().toString(36).slice(2, 10);

export default function Home() {
  const { profile, ready } = useProfile();
  // Chats are namespaced by account so two people sharing a browser do not
  // read each other's history. Signed out, everything lands in the anonymous
  // bucket, which is also what an unconfigured deployment uses.
  const storeKey = profile ? `${STORE_BASE}.${profile.id}` : STORE_BASE;

  const [chats, setChats] = useState<Chat[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [online, setOnline] = useState<boolean | null>(null);
  const [canUpload, setCanUpload] = useState(false);
  // On by default. Asked a factual question without it, this model answers
  // confidently from weights that do not contain the answer -- "the Prime
  // Minister of India" came back as a British citizen who died in 2018. A
  // retrieved passage is not a guarantee either, but it puts the real text on
  // screen under the reply, which nothing else here does.
  const [canWeb, setCanWeb] = useState(false);
  const [web, setWeb] = useState(true);
  const [doc, setDoc] = useState<{ id: string; name: string; chunks: number } | null>(null);
  const [uploading, setUploading] = useState(false);
  const [navOpen, setNavOpen] = useState(false);
  const [copied, setCopied] = useState<number | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  const active = chats.find((c) => c.id === activeId) || null;
  const msgs = active?.msgs ?? [];
  const empty = msgs.length === 0;

  // Chats live in localStorage, not on the server. The backend is a
  // workstation behind a tunnel with no database and no accounts, so keeping
  // history in the browser is the honest place for it -- nothing to leak, and
  // it survives the tunnel going down.
  useEffect(() => {
    if (!ready) return;
    try {
      const raw = localStorage.getItem(storeKey);
      setChats(raw ? JSON.parse(raw) : []);
    } catch {
      /* corrupt or unavailable storage is not worth crashing the page over */
    }
    setActiveId(null);
  }, [storeKey, ready]);

  useEffect(() => {
    if (!ready) return;
    if (chats.length) localStorage.setItem(storeKey, JSON.stringify(chats));
    else localStorage.removeItem(storeKey);
  }, [chats, storeKey, ready]);

  // The backend can go away mid-session, so this polls rather than checking
  // once: a stale "online" badge is worse than no badge.
  useEffect(() => {
    const ping = () =>
      backendUrl().then((api) => fetch(`${api}/health`))
        .then(async (r) => {
          setOnline(r.ok);
          if (r.ok) {
            const h = await r.json();
            setCanUpload(Boolean(h.upload));
            setCanWeb(Boolean(h.web));
          }
        })
        .catch(() => {
          // A dead hostname is the usual cause; drop it so the next poll
          // re-reads backend.json and can recover on its own.
          forgetBackend();
          setOnline(false);
        });
    ping();
    const t = setInterval(ping, 30000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs.length, busy]);

  function patchActive(fn: (m: Msg[]) => Msg[], id?: string) {
    const target = id ?? activeId;
    setChats((cs) =>
      cs.map((c) => (c.id === target ? { ...c, msgs: fn(c.msgs), at: Date.now() } : c))
    );
  }

  async function copy(text: string, i: number) {
    // Wrapped because the Clipboard API is unavailable over plain HTTP and can
    // reject outright on mobile browsers; a failed copy should do nothing
    // rather than throw inside a click handler.
    try {
      await navigator.clipboard?.writeText(text);
      setCopied(i);
      setTimeout(() => setCopied(null), 1400);
    } catch {
      /* no clipboard access -- leave the button as it was */
    }
  }

  function regenerate() {
    // Drop the last exchange and resend the prompt. Sampling is stochastic, so
    // a second attempt on a model this small is often materially better.
    const lastUser = [...msgs].reverse().find((m) => m.role === "user");
    if (!lastUser || busy) return;
    patchActive((m) => m.slice(0, -2));
    send(lastUser.text);
  }

  function grow() {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  }

  function newChat() {
    setActiveId(null);
    setDoc(null);
    setNavOpen(false);
  }

  async function send(text: string) {
    if (!text.trim() || busy) return;
    setInput("");
    setNavOpen(false);
    if (taRef.current) taRef.current.style.height = "auto";

    let id = activeId;
    if (!id) {
      id = newId();
      const title = text.length > 42 ? text.slice(0, 42).trimEnd() + "…" : text;
      setChats((cs) => [{ id: id!, title, at: Date.now(), msgs: [] }, ...cs]);
      setActiveId(id);
    }

    patchActive((m) => [...m, { role: "user", text }, { role: "assistant", text: "" }], id);
    setBusy(true);

    try {
      const res = await fetch(`${await backendUrl()}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          max_tokens: 512,
          temperature: 0.5,
          doc_id: doc?.id ?? null,
          // An uploaded document always wins: the user picked that corpus.
          web: web && canWeb && !doc,
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
          if (d.sources?.length)
            patchActive((m) => {
              const c = [...m];
              c[c.length - 1] = { ...c[c.length - 1], sources: d.sources };
              return c;
            }, id);
          if (d.token)
            patchActive((m) => {
              const c = [...m];
              c[c.length - 1] = { ...c[c.length - 1], text: c[c.length - 1].text + d.token };
              return c;
            }, id);
        }
      }
    } catch {
      patchActive((m) => {
        const c = [...m];
        c[c.length - 1] = {
          role: "assistant",
          text: "Could not reach the model server. It runs on a workstation behind a tunnel and may be offline right now.",
        };
        return c;
      }, id);
    } finally {
      setBusy(false);
    }
  }

  async function upload(f: File) {
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", f);
      const r = await fetch(`${await backendUrl()}/upload`, { method: "POST", body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || "upload failed");
      setDoc({ id: d.doc_id, name: d.name, chunks: d.chunks });
      let id = activeId;
      if (!id) {
        id = newId();
        setChats((cs) => [{ id: id!, title: d.name, at: Date.now(), msgs: [] }, ...cs]);
        setActiveId(id);
      }
      patchActive(
        (m) => [
          ...m,
          {
            role: "assistant",
            text: `Indexed ${d.name} — ${d.chunks} passages. Ask about it and I will answer from the document rather than from memory.`,
          },
        ],
        id
      );
    } catch (e) {
      alert(`Could not read that file. ${(e as Error).message}`);
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className={`shell ${navOpen ? "navopen" : ""}`}>
      <aside>
        <div className="asidetop">
          <div className="brand">
            <span className="mark">स</span>
            <span className="name">Sutra</span>
            <span className="badge">1.3B</span>
          </div>
          <button className="newchat" onClick={newChat}>
            <span>＋</span> New chat
          </button>
        </div>

        <div className="history">
          {chats.length === 0 && <p className="empty">Saved chats appear here</p>}
          {chats.map((c) => (
            <div key={c.id} className={`histrow ${c.id === activeId ? "on" : ""}`}>
              <button
                onClick={() => {
                  setActiveId(c.id);
                  setNavOpen(false);
                }}
              >
                {c.title}
              </button>
              <span
                role="button"
                aria-label="Delete chat"
                onClick={() => {
                  setChats((cs) => cs.filter((x) => x.id !== c.id));
                  if (activeId === c.id) setActiveId(null);
                }}
              >
                ×
              </span>
            </div>
          ))}
        </div>

        <div className="asidefoot">
          <SidebarAccount />
          <span className={`status ${online ? "up" : online === false ? "down" : ""}`}>
            <i /> {online === null ? "checking" : online ? "online" : "offline"}
          </span>
          <div className="links">
            <a href="https://github.com/Abhisingh18/Sutra-1.3B-Model">GitHub</a>
            <a href="https://huggingface.co/Abhisingh-18/Sutra-1.3B-Chat">Weights</a>
          </div>
        </div>
      </aside>

      <div className="pane" onClick={() => navOpen && setNavOpen(false)} />

      <div className="col">
        <header>
          <button className="burger" onClick={() => setNavOpen(!navOpen)} aria-label="Menu">
            ☰
          </button>
          <span className="htitle">{active ? active.title : "New chat"}</span>
        </header>

        <main className={empty ? "centered" : ""}>
          {empty ? (
            <div className="chathero">
              <span className="ornament small" aria-hidden="true">
                ❦
              </span>
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
                    {m.text ? (
                      m.role === "assistant" ? (
                        <Markdown text={m.text} />
                      ) : (
                        m.text
                      )
                    ) : (
                      <span className="dots">
                        <i />
                        <i />
                        <i />
                      </span>
                    )}
                  </div>
                  {m.role === "assistant" && m.text && !busy && (
                    <div className="actions">
                      <button onClick={() => copy(m.text, i)}>
                        {copied === i ? "Copied" : "Copy"}
                      </button>
                      {i === msgs.length - 1 && (
                        <button onClick={regenerate}>Regenerate</button>
                      )}
                    </div>
                  )}
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
            {canWeb && !doc && (
              <button
                className={`tool ${web ? "on" : ""}`}
                onClick={() => setWeb(!web)}
                title="Look the answer up on the web before replying"
              >
                {web ? "✓ " : ""}Search the web
              </button>
            )}
            {doc && (
              <span className="chip">
                {doc.name} · {doc.chunks} passages
                <button onClick={() => setDoc(null)} aria-label="Remove">
                  ×
                </button>
              </span>
            )}
          </div>

          <p className="disclaimer">
            Trained on 18B tokens — about 500x less than comparable 1B models. It
            writes fluently but does not reliably know facts.
          </p>
        </div>
      </div>
    </div>
  );
}
