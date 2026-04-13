import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/layout/Sidebar";
import MobileNav from "@/components/layout/MobileNav";
import FloatingChatWidget from "@/components/chat/FloatingWidget";
import SlicerBar from "@/components/layout/SlicerBar";
import { FilterProvider } from "@/lib/FilterContext";

export const metadata: Metadata = {
  title: "FinTrack — AI Finance Tracker",
  description: "AI-powered personal finance tracker — CS5100 Final Project",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex">
        <FilterProvider>
          <Sidebar />
          <main className="flex-1 md:ml-60 pb-20 md:pb-0">
            <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
              <SlicerBar />
              {children}
            </div>
          </main>
          <MobileNav />
          <FloatingChatWidget />
        </FilterProvider>
      </body>
    </html>
  );
}
