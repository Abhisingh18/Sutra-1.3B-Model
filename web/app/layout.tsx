import type { Metadata } from "next";
import { Fraunces, Inter } from "next/font/google";
import { SessionProvider } from "next-auth/react";

import { ProfileProvider } from "./profile";
import "./globals.css";

/* next/font downloads and self-hosts these at build time, so there is no
   runtime request to Google and no flash of unstyled text. The pairing does
   the work the old system stack could not: a serif with real contrast for
   display, and a neutral sans that stays legible at 13px in the sidebar. */
const display = Fraunces({
  weight: ["400", "500", "600"],
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
});

const sans = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

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
    <html lang="en" className={`${sans.variable} ${display.variable}`}>
      <body>
        {/* Both providers are additive: with neither a session nor a profile
            the app behaves exactly as it did before sign-in existed. */}
        <SessionProvider>
          <ProfileProvider>{children}</ProfileProvider>
        </SessionProvider>
      </body>
    </html>
  );
}
