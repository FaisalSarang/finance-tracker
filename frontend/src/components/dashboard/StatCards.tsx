"use client";

import { DollarSign, TrendingDown, TrendingUp, CreditCard } from "lucide-react";
import Card from "@/components/ui/Card";
import Skeleton from "@/components/ui/Skeleton";
import { formatCurrency } from "@/lib/utils";
import type { Stats } from "@/lib/types";

interface StatCardsProps {
  stats: Stats | undefined;
  loading: boolean;
}

export default function StatCards({ stats, loading }: StatCardsProps) {
  if (loading || !stats) {
    return (
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[1, 2, 3, 4].map((i) => (
          <Card key={i}>
            <Skeleton className="h-4 w-24 mb-3" />
            <Skeleton className="h-8 w-32" />
          </Card>
        ))}
      </div>
    );
  }

  const cards = [
    {
      label: "Total Spent",
      value: formatCurrency(stats.total_spent),
      icon: <TrendingDown size={20} />,
      color: "var(--accent-red)",
      bg: "rgba(248,113,113,0.1)",
    },
    {
      label: "Total Income",
      value: formatCurrency(stats.total_income),
      icon: <TrendingUp size={20} />,
      color: "var(--accent-green)",
      bg: "rgba(126,232,168,0.1)",
    },
    {
      label: "Net Balance",
      value: formatCurrency(stats.net_balance),
      icon: <DollarSign size={20} />,
      color: stats.net_balance >= 0 ? "var(--accent-green)" : "var(--accent-red)",
      bg: stats.net_balance >= 0 ? "rgba(126,232,168,0.1)" : "rgba(248,113,113,0.1)",
    },
    {
      label: "Transactions",
      value: stats.total_transactions.toLocaleString(),
      icon: <CreditCard size={20} />,
      color: "var(--accent-blue)",
      bg: "rgba(88,166,255,0.1)",
    },
  ];

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {cards.map((c) => (
        <Card key={c.label}>
          <div className="flex items-center justify-between">
            <p className="text-sm text-[var(--text-secondary)]">{c.label}</p>
            <div
              className="flex h-9 w-9 items-center justify-center rounded-lg"
              style={{ backgroundColor: c.bg, color: c.color }}
            >
              {c.icon}
            </div>
          </div>
          <p
            className="mt-2 text-2xl font-bold font-mono-numbers"
            style={{ color: c.color }}
          >
            {c.value}
          </p>
        </Card>
      ))}
    </div>
  );
}
