"use client";

import { Trash2, Edit3 } from "lucide-react";
import Badge from "@/components/ui/Badge";
import Skeleton from "@/components/ui/Skeleton";
import { formatCurrency, formatDate, truncate } from "@/lib/utils";
import type { Transaction } from "@/lib/types";

interface TransactionTableProps {
  transactions: Transaction[] | undefined;
  loading: boolean;
  onDelete?: (id: number) => void;
}

export default function TransactionTable({ transactions, loading, onDelete }: TransactionTableProps) {
  if (loading || !transactions) {
    return (
      <div className="space-y-2">
        {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
          <Skeleton key={i} className="h-14 w-full" />
        ))}
      </div>
    );
  }

  if (transactions.length === 0) {
    return (
      <div className="py-16 text-center text-[var(--text-muted)]">
        <p className="text-lg mb-1">No transactions found</p>
        <p className="text-sm">Try adjusting your filters or add new transactions.</p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[var(--border)]">
            <th className="text-left py-3 px-3 text-[var(--text-muted)] font-medium">Date</th>
            <th className="text-left py-3 px-3 text-[var(--text-muted)] font-medium">Description</th>
            <th className="text-left py-3 px-3 text-[var(--text-muted)] font-medium">Category</th>
            <th className="text-right py-3 px-3 text-[var(--text-muted)] font-medium">Amount</th>
            <th className="text-right py-3 px-3 text-[var(--text-muted)] font-medium w-16">Actions</th>
          </tr>
        </thead>
        <tbody>
          {transactions.map((txn) => (
            <tr
              key={txn.id}
              className="border-b border-[var(--border)] border-opacity-50 hover:bg-[var(--bg-surface-hover)] transition-colors"
            >
              <td className="py-3 px-3 text-[var(--text-secondary)] whitespace-nowrap">
                {formatDate(txn.date)}
              </td>
              <td className="py-3 px-3 text-[var(--text-primary)]">
                {truncate(txn.description, 45)}
              </td>
              <td className="py-3 px-3">
                {txn.category && <Badge category={txn.category} />}
              </td>
              <td className="py-3 px-3 text-right font-mono-numbers whitespace-nowrap">
                <span
                  style={{
                    color: txn.transaction_type === "credit" ? "var(--accent-green)" : "var(--accent-red)",
                  }}
                >
                  {txn.transaction_type === "credit" ? "+" : "-"}
                  {formatCurrency(Math.abs(txn.amount))}
                </span>
              </td>
              <td className="py-3 px-3 text-right">
                {onDelete && (
                  <button
                    onClick={() => onDelete(txn.id)}
                    className="rounded-lg p-1.5 text-[var(--text-muted)] hover:text-[var(--accent-red)] hover:bg-[rgba(248,113,113,0.1)] transition-colors"
                    title="Delete"
                  >
                    <Trash2 size={14} />
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
