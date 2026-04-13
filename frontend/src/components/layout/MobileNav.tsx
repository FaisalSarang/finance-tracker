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
} from "lucide-react";
import { cn } from "@/lib/utils";

const ITEMS = [
  { href: "/", icon: <LayoutDashboard size={20} />, label: "Home" },
  { href: "/transactions", icon: <ArrowLeftRight size={20} />, label: "Txns" },
  { href: "/add", icon: <PlusCircle size={22} />, label: "Add" },
  { href: "/analytics", icon: <PieChart size={20} />, label: "Stats" },
  { href: "/chat", icon: <MessageSquare size={20} />, label: "Chat" },
];

export default function MobileNav() {
  const pathname = usePathname();

  return (
    <nav className="fixed bottom-0 left-0 right-0 z-50 border-t border-[var(--border)] bg-[var(--bg-surface)] md:hidden">
      <div className="flex items-center justify-around py-2">
        {ITEMS.map((item) => {
          const isActive =
            pathname === item.href ||
            (item.href !== "/" && pathname.startsWith(item.href));

          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex flex-col items-center gap-0.5 px-3 py-1 text-[10px] font-medium transition-colors",
                isActive
                  ? "text-[var(--accent-green)]"
                  : "text-[var(--text-muted)]"
              )}
            >
              {item.icon}
              {item.label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
