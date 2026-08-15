import type { Metadata } from "next";
import { SessionProvider } from "next-auth/react";

import "./globals.css";

export const metadata: Metadata = {
  title: "Sutra-1.3B",
  description:
    "A 1.32B-parameter Mixture-of-Experts language model trained from scratch on 18B tokens.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        {/* Wraps everything so the chat can read the session on the client.
            Without a session it renders exactly as before -- sign-in is
            additive here, not a gate. */}
        <SessionProvider>{children}</SessionProvider>
      </body>
    </html>
  );
}
