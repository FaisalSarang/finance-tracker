"use client";

import Link from "next/link";
import { ArrowRight } from "lucide-react";
import Card from "@/components/ui/Card";
import ProgressBar from "@/components/ui/ProgressBar";
import Skeleton from "@/components/ui/Skeleton";
import { formatCurrency } from "@/lib/utils";
import type { Budget } from "@/lib/types";

interface BudgetProgressProps {
  budgets: Budget[] | undefined;
  loading: boolean;
}

export default function BudgetProgress({ budgets, loading }: BudgetProgressProps) {
  if (loading || !budgets) {
    return (
      <Card>
        <Skeleton className="h-4 w-36 mb-4" />
        {[1, 2, 3].map((i) => (
          <div key={i} className="mb-4">
            <Skeleton className="h-3 w-24 mb-2" />
            <Skeleton className="h-2 w-full" />
          </div>
        ))}
      </Card>
    );
  }

  return (
    <Card>
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm text-[var(--text-secondary)]">Budget Progress</p>
        <Link
          href="/budgets"
          className="flex items-center gap-1 text-xs text-[var(--accent-blue)] hover:underline"
        >
          Manage <ArrowRight size={12} />
        </Link>
      </div>

      {budgets.length === 0 ? (
        <div className="py-6 text-center text-[var(--text-muted)] text-sm">
          No budgets set.{" "}
          <Link href="/budgets" className="text-[var(--accent-green)] hover:underline">
            Create one
          </Link>
        </div>
      ) : (
        <div className="space-y-4">
          {budgets.slice(0, 4).map((b) => (
            <div key={b.id}>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-sm text-[var(--text-primary)]">{b.category}</span>
                <span className="text-xs text-[var(--text-muted)] font-mono-numbers">
                  {formatCurrency(b.spent)} / {formatCurrency(b.monthly_limit)}
                </span>
              </div>
              <ProgressBar value={b.percentage} />
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}
