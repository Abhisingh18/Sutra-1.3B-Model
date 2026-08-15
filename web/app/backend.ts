/* Where the model server currently lives.
 *
 * A quick tunnel gets a new hostname every time it restarts. Baking that into
 * NEXT_PUBLIC_API_URL means the site points at a dead address until someone
 * redeploys, which is the failure this whole file exists to remove: the
 * supervisor publishes each new hostname to backend.json, and the browser
 * reads it at runtime.
 *
 * Resolution order, and why:
 *   1. backend.json from the deployment -- current, and costs one small fetch
 *   2. NEXT_PUBLIC_API_URL -- whatever was set at build time
 *   3. localhost -- development
 */

const FALLBACK = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

let cached: string | null = null;
let inflight: Promise<string> | null = null;

export function backendUrl(): Promise<string> {
  if (cached) return Promise.resolve(cached);
  if (inflight) return inflight;

  inflight = fetch("/backend.json", { cache: "no-store" })
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => {
      const url = typeof d?.url === "string" && d.url ? d.url : FALLBACK;
      cached = url.replace(/\/+$/, "");
      return cached;
    })
    .catch(() => {
      cached = FALLBACK;
      return cached;
    })
    .finally(() => {
      inflight = null;
    });

  return inflight;
}

/* Forget the resolved address so the next call re-reads backend.json. Called
   when a request fails: the usual cause is that the tunnel restarted and the
   hostname we cached is now dead. */
export function forgetBackend() {
  cached = null;
}
