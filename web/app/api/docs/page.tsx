import Link from "next/link";

/* Public API documentation.
 *
 * The chat page calls /chat with no key -- fine for a browser session, not for
 * someone who wants to call the model from their own code. This page documents
 * the separate /v1 surface that exists for that: a key with no signup, a
 * streaming chat endpoint gated on it, and a usage check. See
 * src/rag/apikeys.py for the limits and why both a per-IP and a per-key cap
 * exist, and deploy/server.py for the routes themselves.
 *
 * A server component, like the other doc pages -- no client JavaScript is
 * needed to show static code blocks.
 */

export default function ApiDocs() {
  return (
    <>
      <header className="topbar">
        <nav className="lnav">
          <Link href="/" className="logo">
            <span className="logomark">स</span>
            sutra<span className="tld">.ai</span>
          </Link>
          <div className="navlinks">
            <Link href="/architecture">Architecture</Link>
            <Link href="/#results">Results</Link>
            <Link href="/api/docs" className="here">
              API
            </Link>
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
          <span className="doceyebrow">Public API</span>
          <h1>Call the model from your own code</h1>
          <p className="doclede">
            No signup, no waitlist. Request a key, send it with your messages,
            and you get the same streaming replies the chat page shows —
            programmatically, from a script or an app of your own.
          </p>
        </header>

        {/* ------------------------------------------------ quickstart */}
        <section className="docsec">
          <span className="seclabel">Quickstart</span>
          <h2>Three requests</h2>
          <p>
            Get a key, spend it on a reply, then check what is left of the
            day&apos;s quota. Replace <code>&lt;base-url&gt;</code> with the
            address on{" "}
            <Link href="/architecture#deployment">the deployment page</Link>{" "}
            — it changes when the tunnel restarts, the same address the chat
            page itself resolves at runtime.
          </p>

          <div className="demo">
            <div className="demobar">
              <i />
              <i />
              <i />
              <span>terminal</span>
            </div>
            <pre className="democode">
              <code>
                <span className="c-mut"># 1. get a key — instant, no signup</span>
                {"\n"}
                <span className="c-mut">$ </span>
                <span className="c-cmd">
                  curl -X POST &lt;base-url&gt;/v1/keys
                </span>
                {"\n\n"}
                <span className="c-dim">
                  {"{"}&quot;api_key&quot;: &quot;sk-sutra-…&quot;,
                  &quot;daily_limit&quot;: 100,{"\n"}
                  {"  "}&quot;note&quot;: &quot;Save this now — it is shown
                  only once…&quot;{"}"}
                </span>
                {"\n\n"}
                <span className="c-mut"># 2. spend it on a reply</span>
                {"\n"}
                <span className="c-mut">$ </span>
                <span className="c-cmd">
                  curl -N -X POST &lt;base-url&gt;/v1/chat \
                </span>
                {"\n  "}
                <span className="c-cmd">
                  -H &quot;Content-Type: application/json&quot; \
                </span>
                {"\n  "}
                <span className="c-cmd">
                  -H &quot;Authorization: Bearer sk-sutra-…&quot; \
                </span>
                {"\n  "}
                <span className="c-cmd">
                  -d &apos;{"{"}&quot;message&quot;: &quot;
                </span>
                <span className="c-str">
                  Explain photosynthesis in one sentence.
                </span>
                <span className="c-cmd">&quot;{"}"}&apos;</span>
                {"\n\n"}
                <span className="c-dim">
                  data: {"{"}&quot;token&quot;: &quot;Photo&quot;{"}"}
                  {"\n"}
                  data: {"{"}&quot;token&quot;: &quot;synthesis&quot;{"}"}
                  {"\n"}
                  data: {"{"}&quot;done&quot;: true{"}"}
                </span>
                {"\n\n"}
                <span className="c-mut"># 3. check what is left today</span>
                {"\n"}
                <span className="c-mut">$ </span>
                <span className="c-cmd">
                  curl &lt;base-url&gt;/v1/usage \
                </span>
                {"\n  "}
                <span className="c-cmd">
                  -H &quot;Authorization: Bearer sk-sutra-…&quot;
                </span>
                {"\n\n"}
                <span className="c-dim">
                  {"{"}&quot;used_today&quot;: 1, &quot;daily_limit&quot;: 100
                  {"}"}
                </span>
              </code>
            </pre>
          </div>
        </section>

        {/* ------------------------------------------------ reference */}
        <section className="docsec">
          <span className="seclabel">Reference</span>
          <h2>Three endpoints</h2>

          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th>Endpoint</th>
                  <th>Auth</th>
                  <th>Does</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>
                    <code>POST /v1/keys</code>
                  </td>
                  <td className="mute">none</td>
                  <td>Mints a new API key. Shown once — it cannot be recovered.</td>
                </tr>
                <tr>
                  <td>
                    <code>POST /v1/chat</code>
                  </td>
                  <td>Bearer key</td>
                  <td>
                    Streams a reply over Server-Sent Events, one <code>token</code>{" "}
                    event at a time, ending in <code>{"{"}&quot;done&quot;: true{"}"}</code>.
                  </td>
                </tr>
                <tr>
                  <td>
                    <code>GET /v1/usage</code>
                  </td>
                  <td>Bearer key</td>
                  <td>Requests spent today against the key&apos;s daily limit.</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3>Request body for /v1/chat</h3>
          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th>Field</th>
                  <th>Type</th>
                  <th>Default</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><code>message</code></td>
                  <td className="mute">string, required</td>
                  <td className="mute">—</td>
                </tr>
                <tr>
                  <td><code>max_tokens</code></td>
                  <td className="mute">int</td>
                  <td className="mute">512</td>
                </tr>
                <tr>
                  <td><code>temperature</code></td>
                  <td className="mute">float</td>
                  <td className="mute">0.5</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* ------------------------------------------------ limits */}
        <section className="docsec">
          <span className="seclabel">Limits</span>
          <h2>Why there are two of them</h2>
          <p>
            This runs on one GPU, shared by the chat page, the demo, and every
            key holder at once. No cap on a self-serve key would let one caller
            take all of it.
          </p>

          <div className="twoup">
            <div>
              <h4>3 keys per IP, per day</h4>
              <p>
                Stops one visitor from minting unlimited keys to route around
                the limit below.
              </p>
            </div>
            <div>
              <h4>100 requests per key, per day</h4>
              <p>
                The actual quota. Resets at 00:00 UTC.{" "}
                <code>GET /v1/usage</code> reports what is left without
                spending a request to find out.
              </p>
            </div>
          </div>

          <div className="panel">
            <p className="panelnote">
              A wrong or missing key gets <code>401</code>. A key over quota
              gets <code>429</code> with the reset time in the message. Both
              come from the same check, atomically with charging the request —
              two calls racing on a key&apos;s last unit of quota cannot both
              slip through.
            </p>
          </div>
        </section>

        <section className="docsec docend">
          <h2>Prefer the chat page?</h2>
          <p>
            <code>/chat</code> needs no key at all — it is the same model,
            with a UI in front of it instead of an SDK.
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
