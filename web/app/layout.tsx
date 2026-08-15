import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sutra-1.3B",
  description:
    "A 1.32B-parameter Mixture-of-Experts language model trained from scratch.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
