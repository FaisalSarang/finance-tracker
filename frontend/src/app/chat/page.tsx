"use client";

import Card from "@/components/ui/Card";
import ChatPanel from "@/components/chat/ChatPanel";
import { Bot } from "lucide-react";

export default function ChatPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-[var(--text-primary)] flex items-center gap-3">
          <Bot size={28} className="text-[var(--accent-green)]" />
          AI Financial Assistant
        </h1>
        <p className="text-sm text-[var(--text-secondary)] mt-1">
          Ask questions about your spending, get financial insights, and receive personalized advice
        </p>
      </div>

      <Card hover={false} className="overflow-hidden">
        <ChatPanel fullPage />
      </Card>
    </div>
  );
}
