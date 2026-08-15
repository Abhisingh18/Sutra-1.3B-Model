import Link from "next/link";

/* Landing page. Deliberately a server component with no client JavaScript --
 * it is static content, and shipping a React bundle to render prose would slow
 * the first thing anyone sees for no gain. The chat app lives at /chat.
 *
 * The numbers below are measured, not marketing: they come from
 * src/eval.py --compare over 500 examples per task. The WinoGrande row sits at
 * chance and is shown anyway, because a benchmark table that only lists wins is
 * not evidence of anything.
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
    body: "48,000-token BPE vocabulary covering English and Devanagari, with chat, reasoning and 4,096 audio tokens reserved up front — adding them later would leave their embeddings untrained.",
  },
  {
    n: "02",
    name: "Pretraining",
    time: "4 days 9 hours",
    body: "18B tokens on 4× RTX 6000 Ada at 33% MFU. Held-out perplexity 15.00, zero dead experts, one loss spike that recovered on its own.",
  },
  {
    n: "03",
    name: "Supervised fine-tuning",
    time: "18 hours",
    body: "200,000 conversations across three epochs. Held-out perplexity 5.49, and the third epoch measured best — so no overfitting.",
  },
  {
    n: "04",
    name: "Preference alignment",
    time: "6 hours",
    body: "DPO over 100,000 preference pairs. Held-out accuracy came out at 47.5% against a 50% baseline, so this stage did not generalise — reported here rather than hidden.",
  },
];

export default function Landing() {
  return (
    <div className="land">
      {/* Full-bleed wash behind the top of the page. Saffron at the crown
          falling through periwinkle into the page background, so the fold has
          weight without any imagery to load. */}
      <div className="wash" aria-hidden="true" />

      <nav className="lnav">
        <div className="brand">
          <span className="mark">स</span>
          <span className="name">Sutra</span>
        </div>
        <div className="lnavlinks">
          <a href="#architecture">Architecture</a>
          <a href="#results">Results</a>
          <a href="https://github.com/Abhisingh18/Sutra-1.3B-Model">GitHub</a>
        </div>
        <div className="navactions">
          <Link href="/chat" className="pill dark">
            Try it
          </Link>
          <Link href="/login" className="pill light">
            Sign in
          </Link>
        </div>
      </nav>

      <header className="lhero">
        <span className="ornament" aria-hidden="true">
          ❦
        </span>
        <span className="eyebrow">
          <span>Trained from scratch in PyTorch</span>
        </span>
        <h1>A language model built from nothing</h1>
        <p>
          1.32 billion parameters across 48 experts, pretrained on 18 billion
          tokens.
          <br />
          Own tokenizer, own data pipeline, own training loop.
        </p>
        <div className="herocta">
          <Link href="/chat" className="pill dark big">
            Start chatting
          </Link>
          <a
            className="pill light big"
            href="https://huggingface.co/Abhisingh-18/Sutra-1.3B-Chat"
          >
            Download weights
          </a>
        </div>
        <dl className="stats">
          <div>
            <dt>1.32B</dt>
            <dd>total parameters</dd>
          </div>
          <div>
            <dt>0.28B</dt>
            <dd>active per token</dd>
          </div>
          <div>
            <dt>18B</dt>
            <dd>training tokens</dd>
          </div>
          <div>
            <dt>4.5</dt>
            <dd>days on 4 GPUs</dd>
          </div>
        </dl>
      </header>

      <section id="architecture" className="lsec">
        <h2>Mixture of Experts, with latent attention</h2>
        <p className="sub">
          Only four of forty-eight experts run for any given token, so the model
          carries 1.32B parameters of capacity at 0.28B parameters of compute.
          That ratio is why it answers on a CPU at roughly ten tokens a second.
        </p>
        <div className="grid3">
          <article>
            <h3>48 + 1 experts</h3>
            <p>
              Top-4 routing with sigmoid scoring and bias-based load balancing.
              One shared expert always fires, absorbing what every token needs so
              the routed ones can specialise. Layer 0 stays dense — routing on
              raw embeddings collapses early.
            </p>
          </article>
          <article>
            <h3>Latent attention</h3>
            <p>
              Multi-head Latent Attention compresses keys and values into a
              256-wide latent before projection, with rotary position handled on
              a decoupled 32-dimension head split.
            </p>
          </article>
          <article>
            <h3>Built to be interrupted</h3>
            <p>
              Checkpoints written atomically, batches a deterministic function of
              step and rank, and a loss-spike guard that rolls back. A four-day
              run does not finish uninterrupted.
            </p>
          </article>
        </div>
      </section>

      <section id="results" className="lsec">
        <h2>Measured, not claimed</h2>
        <p className="sub">
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
          arrived, which is the sharpest available statement of what 0.28B active
          parameters do not buy.
        </p>
      </section>

      <section className="lsec">
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

      <section className="lsec honest">
        <h2>What it cannot do</h2>
        <p className="sub">
          Trained on 18B tokens — roughly 500× less than comparable 1B models.
          That gap shows up in specific, predictable ways, and pretending
          otherwise would waste your time.
        </p>
        <div className="grid2">
          <article className="can">
            <h3>Does well</h3>
            <ul>
              <li>Writes and rewrites — notes, emails, short paragraphs</li>
              <li>Follows formatting instructions: lists, bullets, tone</li>
              <li>Answers from a document you upload</li>
            </ul>
          </article>
          <article className="cant">
            <h3>Does not</h3>
            <ul>
              <li>Recall facts reliably — it states wrong ones confidently</li>
              <li>Reason across multiple steps</li>
              <li>Write working code</li>
              <li>Copy figures accurately, even out of a passage it just read</li>
            </ul>
          </article>
        </div>
      </section>

      <section className="lsec cend">
        <h2>Talk to it</h2>
        <p className="sub">
          Runs on a workstation behind a tunnel. Upload a document and it will
          answer from that, with the passage it used shown underneath.
        </p>
        <Link href="/chat" className="cta">
          Open the chat
        </Link>
      </section>

      <footer className="lfoot">
        <span>Sutra-1.3B · Apache 2.0</span>
        <div>
          <a href="https://github.com/Abhisingh18/Sutra-1.3B-Model">GitHub</a>
          <a href="https://huggingface.co/Abhisingh-18/Sutra-1.3B-Chat">
            Hugging Face
          </a>
        </div>
      </footer>
    </div>
  );
}
