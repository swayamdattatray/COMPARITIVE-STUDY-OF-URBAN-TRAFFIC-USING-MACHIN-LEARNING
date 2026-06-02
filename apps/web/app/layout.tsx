import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CodeMentor AI | Premium AI EdTech Platform",
  description: "Specialized AI trainers, adaptive assessments, coding playground, gamification, and certificates for modern programming education."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body>{children}</body>
    </html>
  );
}
