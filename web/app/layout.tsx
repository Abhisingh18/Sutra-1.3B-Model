import type { Metadata } from "next";

import { ProfileProvider } from "./profile";
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
        {/* Makes the active profile available to the chat. Without one the app
            behaves exactly as before -- a profile is additive, not a gate. */}
        <ProfileProvider>{children}</ProfileProvider>
      </body>
    </html>
  );
}
