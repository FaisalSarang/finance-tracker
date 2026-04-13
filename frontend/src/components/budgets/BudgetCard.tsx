"use client";

import Card from "@/components/ui/Card";
import ProgressBar from "@/components/ui/ProgressBar";
import { formatCurrency } from "@/lib/utils";
import { CATEGORY_CONFIG } from "@/lib/constants";
import type { Budget } from "@/lib/types";

interface BudgetCardProps {
  budget: Budget;
}

export default function BudgetCard({ budget }: BudgetCardProps) {
  const color = CATEGORY_CONFIG[budget.category]?.color ?? "#8B95A8";
  const statusColor =
    budget.percentage > 90 ? "var(--accent-red)" : budget.percentage > 70 ? "var(--accent-yellow)" : "var(--accent-green)";

  return (
    <Card>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full" style={{ backgroundColor: color }} />
          <span className="text-sm font-medium text-[var(--text-primary)]">
            {budget.category}
          </span>
        </div>
        <span className="text-xs font-medium font-mono-numbers" style={{ color: statusColor }}>
          {budget.percentage}%
        </span>
      </div>

      <ProgressBar value={budget.percentage} />

      <div className="flex items-center justify-between mt-3">
        <span className="text-xs text-[var(--text-muted)]">
          Spent: <span className="text-[var(--text-secondary)] font-mono-numbers">{formatCurrency(budget.spent)}</span>
        </span>
        <span className="text-xs text-[var(--text-muted)]">
          Limit: <span className="text-[var(--text-secondary)] font-mono-numbers">{formatCurrency(budget.monthly_limit)}</span>
        </span>
      </div>

      {budget.remaining > 0 ? (
        <p className="mt-2 text-xs text-[var(--accent-green)]">
          {formatCurrency(budget.remaining)} remaining
        </p>
      ) : (
        <p className="mt-2 text-xs text-[var(--accent-red)]">
          Over budget by {formatCurrency(budget.spent - budget.monthly_limit)}
        </p>
      )}
    </Card>
  );
}
