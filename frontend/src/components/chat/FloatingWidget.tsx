"use client";

import { useState } from "react";
import { usePathname } from "next/navigation";
import { MessageSquare, X } from "lucide-react";
import ChatPanel from "./ChatPanel";

export default function FloatingChatWidget() {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();

  // Don't show floating widget on the full chat page
  if (pathname === "/chat") return null;

  return (
    <>
      {/* Chat panel */}
      {open && (
        <div className="fixed bottom-20 right-4 z-50 w-[380px] rounded-2xl border border-[var(--border)] bg-[var(--bg-surface)] shadow-2xl md:bottom-6 md:right-6">
          <div className="flex items-center justify-between border-b border-[var(--border)] px-4 py-3">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-[var(--accent-green)]" />
              <span className="text-sm font-medium text-[var(--text-primary)]">
                AI Assistant
              </span>
            </div>
            <button
              onClick={() => setOpen(false)}
              className="rounded-lg p-1 hover:bg-[var(--bg-surface-hover)] transition-colors"
            >
              <X size={16} className="text-[var(--text-muted)]" />
            </button>
          </div>
          <ChatPanel />
        </div>
      )}

      {/* FAB */}
      <button
        onClick={() => setOpen((prev) => !prev)}
        className="fixed bottom-24 right-4 z-50 flex h-12 w-12 items-center justify-center rounded-full bg-[var(--accent-green)] shadow-lg transition-all duration-300 hover:scale-105 glow-green md:bottom-6 md:right-6"
      >
        {open ? (
          <X size={20} className="text-[#0B0E13]" />
        ) : (
          <MessageSquare size={20} className="text-[#0B0E13]" />
        )}
      </button>
    </>
  );
}
