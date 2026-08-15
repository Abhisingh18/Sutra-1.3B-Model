import Link from "next/link";

/* Landing page. A server component with no client JavaScript -- it is static
 * content, and shipping a React bundle to render prose would slow the first
 * thing anyone sees for no gain. The chat app lives at /chat.
 *
 * Every figure below is measured, not marketing: they come from
 * src/eval.py --compare over 500 examples per task, and src/eval_posttrain.py
 * for the held-out numbers. WinoGrande sits at chance and is shown anyway --
 * a results table that lists only wins is not evidence of anything.
 */

const SCORES = [
  { task: "HellaSwag", random: "25.0", base: "38.4", sft: "39.8", dpo: "40.4" },
  { task: "ARC-easy", random: "25.0", base: "45.0", sft: "44.8", dpo: "45.0" },
  { task: "PIQA", random: "50.0", base: "62.6", sft: "65.4", dpo: "65.6" },
  { task: "WinoGrande", random: "50.0", base: "50.6", sft: "49.0", dpo: "49.0" },
];

const STAGES = [
  {
    n: "01",
    name: "Tokenizer",
    time: "3 hours",
    body: "A 48,000-token BPE vocabulary over English and Devanagari, with chat, reasoning and 4,096 audio tokens reserved up front. Added later, their embeddings would start from noise while everything else had seen 18B tokens.",
  },
  {
    n: "02",
    name: "Pretraining",
    time: "4 days 9 hours",
    body: "18B tokens across 4× RTX 6000 Ada at 33% MFU. Held-out perplexity 15.00, zero dead experts, and one loss spike across 17,166 steps that recovered on its own.",
  },
  {
    n: "03",
    name: "Supervised fine-tuning",
    time: "18 hours",
    body: "200,000 conversations over three epochs. Held-out perplexity 5.49, with the third epoch measuring best — so the extra epochs bought quality rather than overfitting.",
  },
  {
    n: "04",
    name: "Preference alignment",
    time: "6 hours",
    body: "DPO across 100,000 preference pairs. Held-out accuracy landed at 47.5% against a 50% baseline, so this stage did not generalise. Reported here rather than quietly dropped.",
  },
];

export default function Landing() {
  return (
    <>
      {/* A page-level bar pinned to the top edge, above the hero rather than
          floating inside it. */}
      <header className="topbar">
        <nav className="lnav">
          <Link href="/" className="logo">
            <span className="logomark">स</span>
            sutra<span className="tld">.ai</span>
          </Link>
          <div className="navlinks">
            <a href="#architecture">Architecture</a>
            <a href="#results">Results</a>
            <a href="#pipeline">Pipeline</a>
            <a href="https://github.com/Abhisingh18/Sutra-1.3B-Model">Code</a>
          </div>
          <div className="navcta">
            <Link href="/chat" className="btn btn-dark">
              Try it
            </Link>
            <Link href="/login" className="btn btn-ghost">
              Sign in
            </Link>
          </div>
        </nav>
      </header>

      <section className="hero">
        <div className="wrap hero-inner">
          <div className="eyebrow">
            <span className="pipdot" />
            1.32B parameters · 18B tokens · trained on 4 GPUs
          </div>
          <h1>
            Every weight
            <br />
            learned from zero
          </h1>
          <p className="sub">
            A 1.32-billion-parameter Mixture-of-Experts model, written from first
            principles in PyTorch. Own tokenizer, own data pipeline, own training
            loop — nothing pretrained, nothing inherited.
          </p>
          <div className="hero-cta">
            <Link href="/chat" className="btn btn-dark">
              Start chatting
            </Link>
            <a
              className="btn btn-ghost"
              href="https://huggingface.co/Abhisingh-18/Sutra-1.3B-Chat"
            >
              Download weights
            </a>
          </div>

          {/* A fold of pure prose has nothing to look at. This is the shortest
              honest demonstration of the thing: three commands, and what the
              model actually replies. */}
          <div className="demo">
            <div className="demobar">
              <i />
              <i />
              <i />
              <span>terminal</span>
            </div>
            <pre className="democode">
              <code>
                <span className="c-mut">$ </span>
                <span className="c-cmd">pip install torch tokenizers huggingface_hub</span>
                {"\n"}
                <span className="c-mut">$ </span>
                <span className="c-cmd">
                  wget huggingface.co/Abhisingh-18/Sutra-1.3B-Chat/…/inference.py
                </span>
                {"\n"}
                <span className="c-mut">$ </span>
                <span className="c-cmd">python inference.py </span>
                <span className="c-str">&quot;What is machine learning?&quot;</span>
                {"\n\n"}
                <span className="c-dim">
                  Sutra-1.3B on cpu: 1.32B total, 0.28B active
                </span>
                {"\n\n"}
                Machine learning is a branch of artificial intelligence that
                deals with the study and prediction of complex data. It involves
                using algorithms to analyse large amounts of data to make
                predictions about future outcomes.
              </code>
            </pre>
          </div>

          <div className="trust">
            <div className="kicker">Built end to end</div>
            <dl className="logos">
              <div>
                <dt>1.32B</dt>
                <dd>parameters, 48 experts</dd>
              </div>
              <div>
                <dt>0.28B</dt>
                <dd>active per token · 4.7× sparse</dd>
              </div>
              <div>
                <dt>18B</dt>
                <dd>tokens of pretraining</dd>
              </div>
              <div>
                <dt>11</dt>
                <dd>days, tokenizer to aligned</dd>
              </div>
            </dl>
          </div>
        </div>
      </section>

      <section id="architecture" className="platform wrap">
        <h2>Mixture of Experts, with latent attention</h2>
        <p className="lead">
          Four of forty-eight experts run for any given token, so the model
          carries 1.32B parameters of capacity at 0.28B parameters of compute.
          That ratio is why it answers on a CPU at ten tokens a second.
        </p>
        <div className="cards">
          <article className="card">
            <div className="dot" />
            <h3>48 + 1 experts</h3>
            <p>
              Top-4 routing with sigmoid scoring and bias-based load balancing.
              One shared expert always fires, absorbing what every token needs so
              the routed ones can specialise. Layer 0 stays dense — routing on
              raw embeddings collapses early.
            </p>
          </article>
          <article className="card">
            <div className="dot" />
            <h3>Latent attention</h3>
            <p>
              Multi-head Latent Attention compresses keys and values into a
              256-wide latent before projection, with rotary position carried on
              a decoupled 32-dimension head split.
            </p>
          </article>
          <article className="card">
            <div className="dot" />
            <h3>Built to be interrupted</h3>
            <p>
              Atomic checkpoints, batches that are a deterministic function of
              step and rank, and a loss-spike guard that rolls back. A four-day
              run does not finish uninterrupted.
            </p>
          </article>
        </div>
      </section>

      <section id="results" className="platform wrap">
        <h2>Measured, not claimed</h2>
        <p className="lead">
          Log-likelihood scoring over 500 examples per task, length-normalised.
          Reproduce with <code>python -m src.eval --compare</code>.
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
              {SCORES.map((r) => (
                <tr key={r.task}>
                  <td>{r.task}</td>
                  <td className="mute">{r.random}</td>
                  <td>{r.base}</td>
                  <td>{r.sft}</td>
                  <td className="win">{r.dpo}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="note">
          ARC-easy and PIQA sit well clear of chance, so the model learned real
          commonsense rather than fluent grammar alone. WinoGrande sits{" "}
          <em>at</em> chance — the pronoun-resolution reasoning it measures never
          arrived, which is the sharpest statement available of what 0.28B active
          parameters do not buy.
        </p>
      </section>

      <section id="pipeline" className="platform wrap">
        <h2>Four stages, eleven days</h2>
        <ol className="stages">
          {STAGES.map((s) => (
            <li key={s.n}>
              <span className="num">{s.n}</span>
              <div>
                <h3>
                  {s.name} <span className="time">{s.time}</span>
                </h3>
                <p>{s.body}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section className="platform wrap">
        <h2>What it will not do</h2>
        <p className="lead">
          Trained on 18B tokens — roughly 500× less than comparable 1B models.
          That gap shows up in specific, predictable ways, and pretending
          otherwise would only waste your time.
        </p>
        <div className="cards two">
          <article className="card can">
            <h3>Does well</h3>
            <ul>
              <li>Writes and rewrites — notes, emails, short paragraphs</li>
              <li>Follows formatting instructions: lists, bullets, tone</li>
              <li>Answers from a document you upload, and cites the passage</li>
            </ul>
          </article>
          <article className="card cant">
            <h3>Does not</h3>
            <ul>
              <li>Recall facts reliably — it states wrong ones confidently</li>
              <li>Reason across several steps</li>
              <li>Write working code</li>
              <li>Copy figures accurately, even out of a passage it just read</li>
            </ul>
          </article>
        </div>
      </section>

      <section className="band">
        <div className="wrap">
          <h2>Talk to it</h2>
          <p>
            Upload a document and it answers from that, with the passage it used
            shown underneath. No account needed.
          </p>
          <Link href="/chat" className="btn">
            Open the chat
          </Link>
        </div>
      </section>

      <footer className="lfoot">
        <Link href="/" className="logo">
          sutra<span>.ai</span>
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
