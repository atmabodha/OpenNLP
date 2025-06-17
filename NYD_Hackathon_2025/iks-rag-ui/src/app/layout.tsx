import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import LayoutClient from "@/components/Layout/LayoutClient";

export const metadata: Metadata = {
  title: "SAMAY - Your Spiritual Guide",
  description: "Explore spiritual wisdom through modern conversation",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen bg-gray-50">
        <Providers>
          <LayoutClient>{children}</LayoutClient>
        </Providers>
      </body>
    </html>
  );
}
