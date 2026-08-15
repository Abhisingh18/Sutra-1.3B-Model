"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { GoogleButton } from "../authbutton";
import { useProfile } from "../profile";

/* Sign-in portal.
 *
 * Two ways in, both optional. Google follows you to another browser; a local
 * profile needs no setup and no third party. Neither gates the model -- an
 * account only keeps one person's saved chats apart from another's.
 */

export default function LoginClient({
  googleConfigured,
}: {
  googleConfigured: boolean;
}) {
  const { profile, profiles, signIn, switchTo, signOut, remove } = useProfile();
  const [name, setName] = useState("");
  const router = useRouter();

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    signIn(name);
    router.push("/chat");
  }

  return (
    <div className="authwrap">
      {/* Same wash as the landing fold, so signing in does not feel like a
          different product. */}
      <div className="wash auth" aria-hidden="true" />

      <Link href="/" className="backhome">
        &larr; Sutra
      </Link>

      <div className="authcard">
        <span className="mark big">स</span>

        {profile ? (
          <>
            <h1>Signed in as {profile.name}</h1>
            <p>Your saved chats are kept under this profile on this browser.</p>
            <Link href="/chat" className="pill dark wide">
              Go to the chat
            </Link>
            <button
              className="skip asbutton"
              onClick={() => {
                signOut();
                setName("");
              }}
            >
              Switch profile
            </button>
          </>
        ) : (
          <>
            <h1>Who is chatting?</h1>
            <p>
              Sign in to keep your conversations separate from anyone else using
              this browser. The model answers either way — an account is only
              ever about whose chat history is whose.
            </p>

            <GoogleButton configured={googleConfigured} />
            {googleConfigured && <div className="or"><span>or</span></div>}

            {profiles.length > 0 && (
              <div className="profiles">
                {profiles.map((p) => (
                  <div key={p.id} className="prow">
                    <button
                      onClick={() => {
                        switchTo(p.id);
                        router.push("/chat");
                      }}
                    >
                      <span className="avatar" style={{ background: p.colour }}>
                        {p.name.slice(0, 1)}
                      </span>
                      {p.name}
                    </button>
                    <span
                      role="button"
                      aria-label={`Delete ${p.name}`}
                      onClick={() => remove(p.id)}
                    >
                      &times;
                    </span>
                  </div>
                ))}
              </div>
            )}

            <form onSubmit={submit} className="nameform">
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder={profiles.length ? "Add another name" : "Just a name"}
                maxLength={32}
              />
              <button type="submit" className="pill dark" disabled={!name.trim()}>
                Continue
              </button>
            </form>

            <Link href="/chat" className="skip">
              Continue without signing in
            </Link>
          </>
        )}
      </div>

      <p className="authfoot">
        Local profiles never leave this browser. A Google account stores nothing
        but a signed cookie -- there is no database behind this.
      </p>
    </div>
  );
}
