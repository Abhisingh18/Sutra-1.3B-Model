import Link from "next/link";

import { auth, authConfigured, signIn } from "@/auth";

/* Sign-in portal.
 *
 * Signing in is optional throughout this app. It does not gate the model --
 * anyone can chat anonymously -- it only namespaces saved conversations, so
 * two people on one laptop do not read each other's history. Saying that
 * plainly on the page is more useful than the usual "sign in to continue".
 */

export default async function Login() {
  const session = await auth();

  return (
    <div className="authwrap">
      <Link href="/" className="backhome">
        ← Sutra
      </Link>

      <div className="authcard">
        <span className="mark big">स</span>

        {session?.user ? (
          <>
            <h1>You are signed in</h1>
            <p>
              As <strong>{session.user.name || session.user.email}</strong>. Your
              saved chats are kept under this account on this browser.
            </p>
            <Link href="/chat" className="cta wide">
              Go to the chat
            </Link>
          </>
        ) : authConfigured ? (
          <>
            <h1>Sign in to Sutra</h1>
            <p>
              Optional. The model answers without an account — signing in only
              keeps your saved conversations separate from anyone else using
              this browser.
            </p>
            <form
              action={async () => {
                "use server";
                await signIn("google", { redirectTo: "/chat" });
              }}
            >
              <button type="submit" className="gbtn">
                <GoogleMark />
                Continue with Google
              </button>
            </form>
            <Link href="/chat" className="skip">
              Continue without signing in
            </Link>
          </>
        ) : (
          <>
            <h1>Sign-in is not configured</h1>
            <p>
              This deployment has no Google credentials set, so the button is
              hidden rather than shown broken. The chat works without it.
            </p>
            <pre className="envhelp">
              {`# Vercel -> Settings -> Environment Variables
AUTH_GOOGLE_ID=...
AUTH_GOOGLE_SECRET=...
AUTH_SECRET=<openssl rand -base64 32>

# Google Cloud -> Credentials -> OAuth client -> Web
# Authorised redirect URI:
#   https://<your-domain>/api/auth/callback/google`}
            </pre>
            <Link href="/chat" className="cta wide">
              Go to the chat
            </Link>
          </>
        )}
      </div>

      <p className="authfoot">
        No password, no email list. Sessions are a signed cookie; there is no
        database behind this.
      </p>
    </div>
  );
}

function GoogleMark() {
  return (
    <svg width="17" height="17" viewBox="0 0 48 48" aria-hidden="true">
      <path
        fill="#4285F4"
        d="M45.1 24.5c0-1.6-.1-3.1-.4-4.5H24v8.5h11.8c-.5 2.7-2 5-4.4 6.6v5.5h7.1c4.2-3.8 6.6-9.5 6.6-16.1z"
      />
      <path
        fill="#34A853"
        d="M24 46c6 0 11-2 14.6-5.4l-7.1-5.5c-2 1.3-4.5 2.1-7.5 2.1-5.8 0-10.7-3.9-12.4-9.1H4.2v5.7C7.8 41.1 15.3 46 24 46z"
      />
      <path
        fill="#FBBC05"
        d="M11.6 28.1c-.5-1.3-.7-2.7-.7-4.1s.3-2.8.7-4.1v-5.7H4.2C2.8 17.1 2 20.4 2 24s.8 6.9 2.2 9.8l7.4-5.7z"
      />
      <path
        fill="#EA4335"
        d="M24 10.8c3.3 0 6.2 1.1 8.5 3.3l6.3-6.3C35 4.3 30 2 24 2 15.3 2 7.8 6.9 4.2 14.2l7.4 5.7c1.7-5.2 6.6-9.1 12.4-9.1z"
      />
    </svg>
  );
}
