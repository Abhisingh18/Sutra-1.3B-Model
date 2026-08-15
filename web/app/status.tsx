"use client";

import { useEffect, useState } from "react";

import { backendUrl, forgetBackend } from "./backend";

/* Live model status for the landing nav.
 *
 * The backend is a workstation behind a tunnel, so "is it up" is a real
 * question a visitor needs answered before they click through to the chat --
 * finding out by sending a message and getting an error is worse.
 *
 * It polls rather than checking once: the server can go away mid-visit, and a
 * badge that says online because it was online a minute ago is worse than no
 * badge.
 */
export function ModelStatus() {
  const [up, setUp] = useState<boolean | null>(null);

  useEffect(() => {
    let alive = true;
    const ping = () =>
      backendUrl()
        .then((api) => fetch(`${api}/health`, { cache: "no-store" }))
        .then((r) => alive && setUp(r.ok))
        .catch(() => {
          forgetBackend();
          if (alive) setUp(false);
        });
    ping();
    const t = setInterval(ping, 30000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  const label = up === null ? "checking" : up ? "model online" : "model offline";

  return (
    <span
      className={`navstatus ${up ? "up" : up === false ? "down" : ""}`}
      title={
        up === false
          ? "The model runs on a workstation behind a tunnel and is not reachable right now."
          : undefined
      }
    >
      <i />
      {label}
    </span>
  );
}
