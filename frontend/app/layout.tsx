import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Solomind.ai - Clinical Support System",
  description: "Professional AI assistant for pharmacists and medical regulatory professionals.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} antialiased h-screen flex flex-col bg-[#0B1120] text-slate-100`}>
        {children}
      </body>
    </html>
  );
}
