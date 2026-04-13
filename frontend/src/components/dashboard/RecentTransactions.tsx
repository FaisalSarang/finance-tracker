"use client";

import Link from "next/link";
import { ArrowRight } from "lucide-react";
import Card from "@/components/ui/Card";
import Badge from "@/components/ui/Badge";
import Skeleton from "@/components/ui/Skeleton";
import { formatCurrency, formatDate } from "@/lib/utils";
import type { Transaction } from "@/lib/types";

interface RecentTransactionsProps {
  transactions: Transaction[] | undefined;
  loading: boolean;
}

export default function RecentTransactions({ transactions, loading }: RecentTransactionsProps) {
  if (loading || !transactions) {
    return (
      <Card>
        <Skeleton className="h-4 w-44 mb-4" />
        {[1, 2, 3, 4, 5].map((i) => (
          <Skeleton key={i} className="h-12 w-full mb-2" />
        ))}
      </Card>
    );
  }

  return (
    <Card>
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm text-[var(--text-secondary)]">Recent Transactions</p>
        <Link
          href="/transactions"
          className="flex items-center gap-1 text-xs text-[var(--accent-blue)] hover:underline"
        >
          View all <ArrowRight size={12} />
        </Link>
      </div>

      {transactions.length === 0 ? (
        <div className="py-8 text-center text-[var(--text-muted)] text-sm">
          No transactions yet.{" "}
          <Link href="/add" className="text-[var(--accent-green)] hover:underline">
            Add your first
          </Link>
        </div>
      ) : (
        <div className="space-y-1">
          {transactions.slice(0, 7).map((txn) => (
            <div
              key={txn.id}
              className="flex items-center justify-between rounded-lg px-3 py-2.5 hover:bg-[var(--bg-surface-hover)] transition-colors"
            >
              <div className="flex items-center gap-3 min-w-0">
                <div className="min-w-0">
                  <p className="text-sm text-[var(--text-primary)] truncate max-w-[200px]">
                    {txn.description}
                  </p>
                  <p className="text-xs text-[var(--text-muted)]">{formatDate(txn.date)}</p>
                </div>
              </div>
              <div className="flex items-center gap-3 shrink-0">
                {txn.category && <Badge category={txn.category} />}
                <span
                  className="text-sm font-medium font-mono-numbers"
                  style={{
                    color: txn.transaction_type === "credit" ? "var(--accent-green)" : "var(--accent-red)",
                  }}
                >
                  {txn.transaction_type === "credit" ? "+" : "-"}
                  {formatCurrency(Math.abs(txn.amount))}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}
