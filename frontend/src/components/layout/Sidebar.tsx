"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  ArrowLeftRight,
  PlusCircle,
  PieChart,
  Wallet,
  MessageSquare,
  TrendingUp,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { NAV_ITEMS } from "@/lib/constants";

const ICON_MAP: Record<string, React.ReactNode> = {
  LayoutDashboard: <LayoutDashboard size={20} />,
  ArrowLeftRight: <ArrowLeftRight size={20} />,
  PlusCircle: <PlusCircle size={20} />,
  PieChart: <PieChart size={20} />,
  Wallet: <Wallet size={20} />,
  MessageSquare: <MessageSquare size={20} />,
};

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="sidebar-gradient fixed left-0 top-0 z-40 hidden h-screen w-60 border-r border-[var(--border)] md:flex md:flex-col">
      {/* Logo */}
      <div className="flex items-center gap-3 px-6 py-6">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--accent-green)] bg-opacity-15">
          <TrendingUp size={20} className="text-[var(--accent-green)]" />
        </div>
        <span className="text-lg font-bold text-[var(--text-primary)]">
          FinTrack
        </span>
      </div>

      {/* Nav links */}
      <nav className="flex-1 space-y-1 px-3 mt-2">
        {NAV_ITEMS.map((item) => {
          const isActive =
            pathname === item.href ||
            (item.href !== "/" && pathname.startsWith(item.href));

          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200",
                isActive
                  ? "bg-[rgba(126,232,168,0.1)] text-[var(--accent-green)]"
                  : "text-[var(--text-secondary)] hover:bg-[var(--bg-surface-hover)] hover:text-[var(--text-primary)]"
              )}
            >
              {ICON_MAP[item.icon]}
              {item.label}
              {isActive && (
                <span className="ml-auto h-1.5 w-1.5 rounded-full bg-[var(--accent-green)]" />
              )}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="border-t border-[var(--border)] px-6 py-4">
        <p className="text-xs text-[var(--text-muted)]">
          CS5100 Final Project
        </p>
        <p className="text-xs text-[var(--text-muted)]">
          AI Finance Tracker
        </p>
      </div>
    </aside>
  );
}
