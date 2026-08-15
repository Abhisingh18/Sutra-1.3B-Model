"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { useProfile } from "../profile";

/* Sign-in portal.
 *
 * Profiles are local: no server, no password, no email. They exist to keep one
 * person's saved chats apart from another's on a shared browser, which is the
 * only thing an account would buy on a deployment with no database.
 *
 * Google sign-in is not here because it cannot be: OAuth requires a client ID
 * that Google issues to a specific app and redirect URI, so the button would
 * be decorative until those credentials exist.
 */

export default function Login() {
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
      <Link href="/" className="backhome">
        ← Sutra
      </Link>

      <div className="authcard">
        <span className="mark big">स</span>

        {profile ? (
          <>
            <h1>Signed in as {profile.name}</h1>
            <p>
              Your saved chats are kept under this profile on this browser.
            </p>
            <Link href="/chat" className="cta wide">
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
              Pick a name to keep your conversations separate from anyone else
              using this browser. No password, no email — nothing leaves this
              machine.
            </p>

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
                      ×
                    </span>
                  </div>
                ))}
              </div>
            )}

            <form onSubmit={submit} className="nameform">
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder={profiles.length ? "Add another name" : "Your name"}
                maxLength={32}
                autoFocus
              />
              <button type="submit" className="cta" disabled={!name.trim()}>
                Continue
              </button>
            </form>

            <Link href="/chat" className="skip">
              Continue without a profile
            </Link>
          </>
        )}
      </div>

      <p className="authfoot">
        Profiles are stored in this browser only. Clearing site data removes
        them, and the chats with them.
      </p>
    </div>
  );
}
