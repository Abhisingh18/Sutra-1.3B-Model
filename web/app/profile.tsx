"use client";

import { createContext, useContext, useEffect, useState } from "react";

/* Local profiles.
 *
 * Google sign-in needs a client ID issued by Google to a specific app and
 * redirect URI, so it cannot work until those credentials exist. This does the
 * job that sign-in actually does here -- keeping one person's saved chats
 * apart from another's on a shared browser -- and it works the moment the page
 * loads, with nothing to configure.
 *
 * It is not authentication and does not pretend to be: there is no server, no
 * password and no secret. Anyone at this browser can switch profiles, which is
 * the correct amount of security for chats that never leave the machine.
 */

export type Profile = { id: string; name: string; colour: string };

const KEY = "sutra.profile.v1";
const LIST = "sutra.profiles.v1";

// Picked to stay legible against both the ivory and the charcoal background.
const COLOURS = ["#b5451b", "#7c6f1f", "#1f6b52", "#3a5ea8", "#7a3d7c", "#a03d3d"];

type Ctx = {
  profile: Profile | null;
  profiles: Profile[];
  ready: boolean;
  signIn: (name: string) => void;
  switchTo: (id: string) => void;
  signOut: () => void;
  remove: (id: string) => void;
};

const ProfileCtx = createContext<Ctx | null>(null);

export function useProfile() {
  const c = useContext(ProfileCtx);
  if (!c) throw new Error("useProfile outside ProfileProvider");
  return c;
}

export function ProfileProvider({ children }: { children: React.ReactNode }) {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  // `ready` exists so consumers do not read an empty profile during the first
  // render and wipe the chat list they are about to load.
  const [ready, setReady] = useState(false);

  useEffect(() => {
    try {
      const all = JSON.parse(localStorage.getItem(LIST) || "[]") as Profile[];
      const id = localStorage.getItem(KEY);
      setProfiles(all);
      setProfile(all.find((p) => p.id === id) || null);
    } catch {
      /* unreadable storage should not take the page down */
    }
    setReady(true);
  }, []);

  function persist(all: Profile[], current: Profile | null) {
    setProfiles(all);
    setProfile(current);
    localStorage.setItem(LIST, JSON.stringify(all));
    if (current) localStorage.setItem(KEY, current.id);
    else localStorage.removeItem(KEY);
  }

  return (
    <ProfileCtx.Provider
      value={{
        profile,
        profiles,
        ready,
        signIn(name) {
          const clean = name.trim().slice(0, 32);
          if (!clean) return;
          // Reuse an existing profile with the same name rather than making a
          // second one, so signing back in returns you to your own chats.
          const found = profiles.find(
            (p) => p.name.toLowerCase() === clean.toLowerCase()
          );
          if (found) return persist(profiles, found);
          const p: Profile = {
            id: Math.random().toString(36).slice(2, 10),
            name: clean,
            colour: COLOURS[profiles.length % COLOURS.length],
          };
          persist([...profiles, p], p);
        },
        switchTo(id) {
          persist(profiles, profiles.find((p) => p.id === id) || null);
        },
        signOut() {
          persist(profiles, null);
        },
        remove(id) {
          localStorage.removeItem(`sutra.chats.v1.${id}`);
          persist(
            profiles.filter((p) => p.id !== id),
            profile?.id === id ? null : profile
          );
        },
      }}
    >
      {children}
    </ProfileCtx.Provider>
  );
}
