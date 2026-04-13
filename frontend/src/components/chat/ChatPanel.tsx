"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Loader2 } from "lucide-react";
import { chat } from "@/lib/api";
import type { ChatMessage } from "@/lib/types";

interface ChatPanelProps {
  fullPage?: boolean;
}

export default function ChatPanel({ fullPage = false }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    const q = input.trim();
    if (!q || loading) return;

    const userMsg: ChatMessage = { role: "user", content: q };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const { reply } = await chat(q, messages);
      setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Sorry, I couldn't connect to the AI service. Make sure the backend and Ollama are running." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className={`flex flex-col ${fullPage ? "h-[calc(100vh-120px)]" : "h-[420px]"}`}>
      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-4 p-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center gap-3">
            <div className="h-12 w-12 rounded-full bg-[rgba(126,232,168,0.1)] flex items-center justify-center">
              <Bot size={24} className="text-[var(--accent-green)]" />
            </div>
            <p className="text-[var(--text-secondary)] text-sm max-w-xs">
              Ask me about your spending, transactions, budgets, or financial advice based on your data.
            </p>
            <div className="flex flex-wrap gap-2 mt-2">
              {["How much did I spend this month?", "What's my biggest category?", "Any saving tips?"].map((q) => (
                <button
                  key={q}
                  onClick={() => { setInput(q); }}
                  className="text-xs rounded-full border border-[var(--border)] px-3 py-1.5 text-[var(--text-secondary)] hover:border-[var(--accent-green)] hover:text-[var(--accent-green)] transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            {msg.role === "assistant" && (
              <div className="h-7 w-7 rounded-full bg-[rgba(126,232,168,0.1)] flex items-center justify-center shrink-0 mt-0.5">
                <Bot size={14} className="text-[var(--accent-green)]" />
              </div>
            )}
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-[var(--accent-green)] text-[#0B0E13] rounded-br-md"
                  : "bg-[var(--bg-surface-hover)] text-[var(--text-primary)] rounded-bl-md"
              }`}
            >
              {msg.content}
            </div>
            {msg.role === "user" && (
              <div className="h-7 w-7 rounded-full bg-[rgba(88,166,255,0.15)] flex items-center justify-center shrink-0 mt-0.5">
                <User size={14} className="text-[var(--accent-blue)]" />
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex gap-3">
            <div className="h-7 w-7 rounded-full bg-[rgba(126,232,168,0.1)] flex items-center justify-center shrink-0">
              <Bot size={14} className="text-[var(--accent-green)]" />
            </div>
            <div className="bg-[var(--bg-surface-hover)] rounded-2xl rounded-bl-md px-4 py-3">
              <Loader2 size={16} className="animate-spin text-[var(--text-muted)]" />
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t border-[var(--border)] p-3">
        <div className="flex items-center gap-2 rounded-xl bg-[var(--bg-surface-hover)] px-4 py-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about your finances..."
            className="flex-1 bg-transparent text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] outline-none"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || loading}
            className="h-8 w-8 rounded-lg bg-[var(--accent-green)] flex items-center justify-center disabled:opacity-30 transition-opacity hover:opacity-90"
          >
            <Send size={14} className="text-[#0B0E13]" />
          </button>
        </div>
      </div>
    </div>
  );
}
