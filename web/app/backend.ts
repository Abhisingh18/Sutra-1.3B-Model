/* Where the model server currently lives.
 *
 * A quick tunnel gets a new hostname every time it restarts. Baking that into
 * NEXT_PUBLIC_API_URL means the site points at a dead address until someone
 * redeploys, which is the failure this file exists to remove: the supervisor
 * publishes each new hostname to backend.json, and the browser reads it at
 * runtime.
 *
 * Resolution order, and why:
 *   1. backend.json from the deployment -- current, and costs one small fetch
 *   2. NEXT_PUBLIC_API_URL -- whatever was set at build time
 *   3. localhost -- development
 */

const FALLBACK = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/* Read straight from the repo rather than from this deployment. The file in
 * web/public only changes when Vercel rebuilds, so a tunnel restart left the
 * site pointing at a dead hostname for as long as the build took. The raw
 * endpoint updates the moment the supervisor commits, and serves
 * Access-Control-Allow-Origin: * so the browser can read it. */
const LIVE =
  "https://raw.githubusercontent.com/Abhisingh18/Sutra-1.3B-Model/main/web/public/backend.json";

let cached: string | null = null;
let inflight: Promise<string> | null = null;

function clean(url: string): string {
  return url.replace(/\/+$/, "");
}

export function backendUrl(): Promise<string> {
  if (cached) return Promise.resolve(cached);
  if (inflight) return inflight;

  // Resolved through a local so the promise stays Promise<string>; returning
  // the nullable `cached` widens it and the build rejects it.
  inflight = fetch(LIVE, { cache: "no-store" })
    .catch(() => fetch("/backend.json", { cache: "no-store" }))
    .then((r) => (r && r.ok ? r.json() : null))
    .then((d: { url?: unknown } | null) => {
      const found =
        d && typeof d.url === "string" && d.url ? clean(d.url) : clean(FALLBACK);
      cached = found;
      return found;
    })
    .catch(() => {
      const found = clean(FALLBACK);
      cached = found;
      return found;
    })
    .finally(() => {
      inflight = null;
    });

  return inflight;
}

/* Forget the resolved address so the next call re-reads backend.json. Called
   when a request fails: the usual cause is that the tunnel restarted and the
   hostname we cached is now dead. */
export function forgetBackend(): void {
  cached = null;
}
