"use client";

import { signIn, signOut, useSession } from "next-auth/react";

import { useProfile } from "./profile";

/* Google sign-in on top of local profiles.
 *
 * Both exist because they solve the same problem at different strengths. A
 * Google account follows you to another browser; a local profile needs no
 * setup and no third party. Whichever is present namespaces the saved chats,
 * and neither is required to use the model.
 */

export function GoogleButton({ configured }: { configured: boolean }) {
  const { data: session } = useSession();

  if (!configured) return null;

  if (session?.user) {
    return (
      <div className="signedin">
        {session.user.image ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={session.user.image} alt="" />
        ) : null}
        <span>{session.user.name || session.user.email}</span>
        <button onClick={() => signOut({ callbackUrl: "/login" })}>
          Sign out
        </button>
      </div>
    );
  }

  return (
    <button
      className="gbtn"
      onClick={() => signIn("google", { callbackUrl: "/chat" })}
    >
      <GoogleMark />
      Continue with Google
    </button>
  );
}

/* Shown in the chat sidebar: Google account if there is one, otherwise the
   local profile, otherwise a prompt to make one. */
export function SidebarAccount() {
  const { data: session } = useSession();
  const { profile } = useProfile();

  if (session?.user) {
    return (
      <div className="account">
        {session.user.image ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={session.user.image} alt="" />
        ) : (
          <span className="avatar">{(session.user.name || "?").slice(0, 1)}</span>
        )}
        <span className="who2">{session.user.name || session.user.email}</span>
        <button onClick={() => signOut({ callbackUrl: "/login" })}>Out</button>
      </div>
    );
  }

  if (profile) {
    return (
      <div className="account">
        <span className="avatar" style={{ background: profile.colour }}>
          {profile.name.slice(0, 1)}
        </span>
        <span className="who2">{profile.name}</span>
        <a href="/login">Switch</a>
      </div>
    );
  }

  return (
    <a className="signin" href="/login">
      Sign in to keep chats separate
    </a>
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
