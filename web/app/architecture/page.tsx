import Link from "next/link";

import { ExpertGrid } from "./grid";

/* The architecture page.
 *
 * A server component with no client JavaScript except the expert grid, which
 * is generated rather than hand-authored -- 48 identical rects differing only
 * in position is exactly the kind of SVG that should be drawn in a loop.
 *
 * Every figure here is measured: durations and perplexities come from
 * logs_train.txt and logs_sft.txt, benchmark scores from src/eval.py, the
 * parameter ledger from MoEModelConfig.param_count().
 */

const STAGES = [
  { n: "tokenizer", time: "3 hours", out: "tokenizer.json", metric: "48,000 vocab", key: true },
  { n: "data prep", time: "~1 day", out: "98 shards", metric: "49 GB on disk" },
  { n: "pretrain", time: "4d 9h", out: "final.pt", metric: "ppl 15.00", key: true },
  { n: "SFT", time: "18 hours", out: "sft_epoch_2.pt", metric: "ppl 5.49" },
  { n: "DPO", time: "6 hours", out: "dpo_epoch_0.pt", metric: "acc 47.5%" },
  { n: "serve", time: "continuous", out: "public URL", metric: "10 tok/s CPU" },
];

const BUGS = [
  {
    tag: "Pretraining",
    title: "DDP hangs on a sparse model",
    body: "Top-4 routing over 48 experts leaves some experts with no tokens in a micro-batch, so their gradients never arrive and their DDP buckets never become ready. Different ranks skip different experts, so the collectives desync. find_unused_parameters=True is not optional here — and it was present in train.py but missing from sft.py and dpo.py, which is where it surfaced.",
  },
  {
    tag: "Fine-tuning",
    title: "DPO forwards twice, backwards once",
    body: "DPO scores a chosen and a rejected answer before a single backward pass. DDP marks each parameter ready when its gradient arrives, so two forwards into one backward marks everything twice and aborts. The fix is what the reference implementations do: concatenate both candidates along the batch axis and forward once.",
  },
  {
    tag: "Generation",
    title: "“What is AI?” → “AI”",
    body: "The model emitted <|end_turn|> three tokens in with probability 0.83, so every reply was a fragment. Blocking the stop token for the first 32 steps forces it to elaborate. Separately, top_p was accepted as an argument and never applied — every reply was really plain top-k 50.",
  },
  {
    tag: "Retrieval",
    title: "Wikipedia recall stuck at exactly 50%",
    body: "Two real bugs, and neither was the cause. The dump streams alphabetically, so the first 500k articles were A–C: Albert Einstein was there, Tokyo was not. Shuffling fixed that and the number did not move, because the median article is ~1,300 characters and a random sample is almost all stubs. What remained is arithmetic — any few hundred thousand articles is 2–6% of Wikipedia. Retrieval over uploaded documents shipped instead, where coverage is total by construction.",
  },
  {
    tag: "Frontend",
    title: "A bare CSS selector moved the navigation",
    body: "The chat app styled header, main and aside as bare element selectors, and the stylesheet is shared. header { display: flex } turned the landing page's header into a row and laid the nav and the hero side by side, vertically centred, halfway down the page.",
  },
  {
    tag: "Serving",
    title: "A live process is not a working service",
    body: "The supervisor restarted the server forever while health checks passed: an orphan held port 8000, so every new server exited with EADDRINUSE while the orphan answered the probe. Health checks also ran ten seconds after start, killing each server mid-load of its 5.3 GB of weights. And a quick tunnel can hang after its preflight without ever registering — a live process behind a hostname that resolves to nothing.",
  },
];

export default function Architecture() {
  return (
    <>
      <header className="topbar">
        <nav className="lnav">
          <Link href="/" className="logo">
            <span className="logomark">स</span>
            sutra<span className="tld">.ai</span>
          </Link>
          <div className="navlinks">
            <Link href="/architecture" className="here">
              Architecture
            </Link>
            <Link href="/#results">Results</Link>
            <a href="https://github.com/Abhisingh18/Sutra-1.3B-Model">Code</a>
          </div>
          <div className="navcta">
            <Link href="/chat" className="btn btn-dark">
              Try it
            </Link>
          </div>
        </nav>
      </header>

      <div className="doc wrap">
        <header className="dochead">
          <span className="doceyebrow">Architecture &amp; workflow</span>
          <h1>How it was built</h1>
          <p className="doclede">
            A Mixture-of-Experts language model built end to end in PyTorch —
            own tokenizer, own data pipeline, own training loop. Nothing
            pretrained, nothing inherited. This is the whole system, stage by
            stage, with the numbers each stage actually produced.
          </p>
        </header>

        {/* ------------------------------------------------ pipeline */}
        <section className="docsec">
          <span className="seclabel">The spine</span>
          <h2>Six stages</h2>
          <p>
            Each stage consumes the output of the last, and can only run once
            the one before it has finished. Two are irreversible: the tokenizer
            fixes the vocabulary every later stage depends on, and pretraining
            fixes the weights everything after it merely adjusts.
          </p>

          <ol className="pipeline">
            {STAGES.map((s) => (
              <li key={s.n} className={s.key ? "key" : ""}>
                <span className="pname">{s.n}</span>
                <span className="ptime">{s.time}</span>
                <span className="pout">{s.out}</span>
                <span className="pmetric">{s.metric}</span>
                {s.key && <span className="plock">irreversible</span>}
              </li>
            ))}
          </ol>
        </section>

        {/* ------------------------------------------------ tokenizer */}
        <section className="docsec">
          <span className="seclabel">Stage 01 · run once, changes everything</span>
          <h2>Tokenizer</h2>
          <p>
            Byte-level BPE over a 20 GB sample drawn from the same mixture the
            model will train on. Starting from raw bytes, the most frequent
            adjacent pair is merged into a new token, 43,861 times over.
          </p>
          <p>
            Hindi is sampled at <b>twice its share of the training mix</b>.
            Devanagari needs more merges than Latin to encode efficiently, and
            an under-merged Hindi vocabulary silently doubles the token cost of
            every Hindi document for the entire run — a mistake you would only
            discover four days in.
          </p>

          <div className="panel">
            <div className="tokrow">
              <div>
                <span className="tokline">Machine learning is useful.</span>
                <div className="chips">
                  {["Machine", "·learning", "·is", "·useful", "."].map((t) => (
                    <span key={t} className="chip">
                      {t}
                    </span>
                  ))}
                </div>
              </div>
              <span className="tokcount">5 tokens · 27 chars</span>
            </div>

            <div className="tokrow">
              <div>
                <span className="tokline">मशीन लर्निंग उपयोगी है।</span>
                <div className="chips">
                  {Array.from({ length: 14 }).map((_, i) => (
                    <span key={i} className="chip blank" />
                  ))}
                </div>
              </div>
              <span className="tokcount hot">14 tokens · 23 chars</span>
            </div>

            <p className="panelnote">
              Same meaning, 2.8× the tokens. That ratio is the price of a 57%
              English mixture — and it is fixed for the life of the model.
            </p>

            <div className="vocabbar">
              <span className="learned">43,861 learned merges</span>
              <span className="reserved">4,139 reserved</span>
            </div>
            <p className="panelnote">
              The reserved block — chat, reasoning, tool-use, 32 spares and 4,096
              audio slots — costs 0.7% of the model and had to exist before
              pretraining. Added later, those embeddings start from noise while
              everything else has seen 18B tokens, and never catch up.
            </p>
          </div>
        </section>

        {/* ------------------------------------------------ data */}
        <section className="docsec">
          <span className="seclabel">Stage 02 · two phases</span>
          <h2>Data preparation</h2>
          <p>
            Raw text is never written to disk. It streams from HuggingFace, is
            tokenized in flight, and only <code>uint16</code> ids are stored —
            turning a ~500 GB text problem into a 49 GB one.
          </p>

          <div className="phases">
            <div className="phase">
              <span className="phlabel">Phase 1 — fetch, one source at a time</span>
              <ul className="sources">
                <li><span>FineWeb-Edu</span><b>57%</b></li>
                <li><span>Sangraha (Hindi)</span><b>12%</b></li>
                <li><span>CodeParrot + notebooks</span><b>12%</b></li>
                <li><span>open-web-math, FineMath</span><b>9%</b></li>
                <li><span>Gutenberg, arXiv, Wikipedia</span><b>10%</b></li>
              </ul>
              <p className="phnote">
                Each becomes its own <code>.bin</code> of uint16 ids, with an EOS
                after every document. Batches are capped by <b>characters, not
                document count</b> — document sizes span four orders of magnitude
                here, and a fixed 2,000-document batch of Gutenberg is ~225M
                tokens, enough to overshoot a 2M quota a hundredfold.
              </p>
            </div>

            <div className="phase">
              <span className="phlabel">Phase 2 — interleave into shards</span>
              <div className="shards">
                {["00000", "00001", "00002", "00003", "…097"].map((s) => (
                  <span key={s} className="shard">
                    {s}
                  </span>
                ))}
              </div>
              <p className="phnote">
                Every shard carries the whole mixture. Without interleaving the
                model sees three days of English, then a day of Hindi — and
                forgets as it goes. 98 shards, 250M tokens each, 24.5B total.
              </p>
            </div>
          </div>
        </section>

        {/* ------------------------------------------------ model */}
        <section className="docsec">
          <span className="seclabel">The model</span>
          <h2>16 layers, 49 experts each</h2>
          <p>
            Every token enters as one of 48,000 ids, becomes a
            1024-dimensional vector, and passes through 16 blocks. Each block
            does two things: attention, which looks at other tokens, and a
            feed-forward network, which does the actual work.
          </p>

          <div className="ledger">
            <span className="ledgerhead">Where the parameters sit</span>
            {[
              { n: "embedding", total: 49.2, active: 49.2, w: 4 },
              { n: "attention ×16", total: 54.4, active: 54.4, w: 4 },
              { n: "dense FFN, layer 0", total: 12.1, active: 12.1, w: 1 },
              { n: "MoE FFN ×15", total: 1208.4, active: 115, w: 100 },
            ].map((r) => (
              <div key={r.n} className="lrow">
                <span className="lname">{r.n}</span>
                <span className="lbar">
                  <span className="ltotal" style={{ width: `${r.w}%` }}>
                    <span
                      className="lactive"
                      style={{ width: `${(r.active / r.total) * 100}%` }}
                    />
                  </span>
                </span>
                <span className="lnum">
                  {r.total.toLocaleString()}M<i> / {r.active.toLocaleString()}M</i>
                </span>
              </div>
            ))}
            <p className="panelnote">
              Filled is what runs for a given token; the outline is what is
              stored. Almost all of the model is MoE weight, and almost none of
              it fires — 1,318.8M total against 280.0M active, 4.71× sparse.
            </p>
          </div>

          <h3>Routing: 49 sit, 5 fire</h3>
          <p>
            This is the whole idea. Each MoE layer holds 48 routed experts plus
            one shared expert. A router scores the 48 and picks the best four;
            the shared expert always runs. The other 44 are not touched for that
            token.
          </p>

          <ExpertGrid />

          <div className="twoup">
            <div>
              <h4>Why a shared expert</h4>
              <p>
                It absorbs what every token needs — basic grammar, common
                patterns — so the routed experts are free to specialise instead
                of each having to be a generalist.
              </p>
            </div>
            <div>
              <h4>Why layer 0 stays dense</h4>
              <p>
                Routing on raw embeddings is near-random, because the model has
                not learned anything yet. A router that collapses in the first
                layer never recovers: a few experts take every token and the
                rest never train.
              </p>
            </div>
            <div>
              <h4>Why bias, not an auxiliary loss</h4>
              <p>
                Each expert carries a bias added to its routing score;
                overloaded experts get nudged down, idle ones up. It happens
                outside the loss, so balance costs the objective nothing. An
                auxiliary loss fights the thing you are training for.
              </p>
            </div>
            <div>
              <h4>What it measured</h4>
              <p>
                Load stayed between 1.9% and 2.4% against a 2.1% uniform, with{" "}
                <b>zero dead experts</b> across all 17,166 steps. Collapse is
                the failure mode this architecture is most exposed to, and it
                did not happen.
              </p>
            </div>
          </div>
        </section>

        {/* ------------------------------------------------ results */}
        <section className="docsec">
          <span className="seclabel">Evaluation</span>
          <h2>Measured, including what failed</h2>
          <p>
            Log-likelihood scoring, 500 examples per task, length-normalised. A
            results table that lists only wins is not evidence, so the two that
            went the other way are here too.
          </p>

          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Random</th>
                  <th>Base</th>
                  <th>SFT</th>
                  <th>DPO</th>
                </tr>
              </thead>
              <tbody>
                <tr><td>HellaSwag</td><td className="mute">25.0</td><td>38.4</td><td>39.8</td><td className="win">40.4</td></tr>
                <tr><td>ARC-easy</td><td className="mute">25.0</td><td>45.0</td><td>44.8</td><td className="win">45.0</td></tr>
                <tr><td>PIQA</td><td className="mute">50.0</td><td>62.6</td><td>65.4</td><td className="win">65.6</td></tr>
                <tr><td>WinoGrande</td><td className="mute">50.0</td><td>50.6</td><td className="mute">49.0</td><td className="mute">49.0</td></tr>
              </tbody>
            </table>
          </div>

          <p>
            ARC-easy and PIQA sit well clear of chance, so the model learned
            real commonsense rather than fluent grammar alone.{" "}
            <b>WinoGrande sits at chance</b> — the pronoun-resolution reasoning
            it measures never arrived, which is the sharpest statement available
            of what 0.28B active parameters do not buy.
          </p>
          <p>
            <b>DPO did not generalise.</b> Its training loop reported 66.25%
            preference accuracy; on held-out pairs it came out at 47.5% against
            a 50% baseline, and the benchmark columns agree — SFT and DPO are
            within noise of each other. The stage cost six hours and bought
            nothing measurable. It went unnoticed until afterwards because
            neither <code>sft.py</code> nor <code>dpo.py</code> had any
            validation at all.
          </p>
        </section>

        {/* ------------------------------------------------ serving */}
        <section className="docsec">
          <span className="seclabel">Deployment</span>
          <h2>Getting it online, and keeping it there</h2>
          <p>
            The model runs on the GPU box; this site runs on Vercel. The hard
            part is not connecting them — it is that the tunnel between them
            gets a new hostname every time it restarts.
          </p>

          <div className="chain">
            <span className="node">browser</span>
            <span className="arrow" />
            <span className="node">Vercel</span>
            <span className="arrow" />
            <span className="node">cloudflared</span>
            <span className="arrow" />
            <span className="node hot">FastAPI + model</span>
          </div>

          <div className="panel">
            <p className="panelnote">
              <b>The address is published, not configured.</b> The supervisor
              writes each new hostname to <code>backend.json</code> in the repo,
              and the browser reads it at runtime from raw.githubusercontent.com.
              Baking it into an environment variable meant a redeploy after every
              restart — and the redeploy is triggered by the very commit carrying
              the new address, so the gap was guaranteed. When a request fails
              the cached address is dropped and the next poll re-reads the file,
              so the site recovers on its own.
            </p>
            <p className="panelnote">
              <b>Two layers of supervision</b>, each covering what the other
              cannot. <code>serve.sh</code> health-checks the model server and
              the public URL, restarting either within ten seconds. systemd
              watches <code>serve.sh</code> — the one thing it cannot watch
              itself — and lingering keeps it alive across logout and reboot.
            </p>
          </div>
        </section>

        {/* ------------------------------------------------ bugs */}
        <section className="docsec">
          <span className="seclabel">What broke</span>
          <h2>Six bugs worth keeping</h2>
          <p>
            Every one of these was silent — the code ran, produced output, and
            was wrong. They are the part of the project most likely to be useful
            to somebody else.
          </p>

          <div className="bugs">
            {BUGS.map((b) => (
              <article key={b.title} className="bug">
                <span className="bugtag">{b.tag}</span>
                <h4>{b.title}</h4>
                <p>{b.body}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="docsec docend">
          <h2>Talk to it</h2>
          <p>
            Upload a document and it answers from that, with the passage it used
            shown underneath. No account needed.
          </p>
          <Link href="/chat" className="btn btn-dark">
            Open the chat
          </Link>
        </section>
      </div>

      <footer className="lfoot">
        <Link href="/" className="logo">
          <span className="logomark">स</span>
          sutra<span className="tld">.ai</span>
        </Link>
        <small>
          1.32B Mixture-of-Experts · Apache 2.0 ·{" "}
          <a href="https://github.com/Abhisingh18/Sutra-1.3B-Model">GitHub</a> ·{" "}
          <a href="https://huggingface.co/Abhisingh-18/Sutra-1.3B-Chat">
            Hugging Face
          </a>
        </small>
      </footer>
    </>
  );
}
