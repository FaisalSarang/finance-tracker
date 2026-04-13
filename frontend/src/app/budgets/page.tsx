"use client";

import useSWR from "swr";
import { getBudgets } from "@/lib/api";
import { useFilters } from "@/lib/FilterContext";
import { formatMonth } from "@/lib/utils";
import Card from "@/components/ui/Card";
import Skeleton from "@/components/ui/Skeleton";
import BudgetCard from "@/components/budgets/BudgetCard";
import BudgetForm from "@/components/budgets/BudgetForm";

export default function BudgetsPage() {
  const { filters, budgetMonth } = useFilters();

  const { data: budgets, isLoading, mutate } = useSWR(
    ["budgets", budgetMonth],
    () => getBudgets(budgetMonth)
  );

  // If a category slicer is active, show only that category's budget
  const displayBudgets = budgets && filters.category
    ? budgets.filter((b) => b.category === filters.category)
    : budgets;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-[var(--text-primary)]">Budgets</h1>
        <p className="text-sm text-[var(--text-secondary)] mt-1">
          Set spending limits by category
          {filters.category && <span className="text-[var(--accent-blue)]"> — {filters.category}</span>}
        </p>
      </div>

      {/* Create budget form */}
      <Card hover={false}>
        <p className="text-sm text-[var(--text-secondary)] mb-3">
          Add a new budget for {formatMonth(budgetMonth)}
        </p>
        <BudgetForm month={budgetMonth} onCreated={() => mutate()} />
      </Card>

      {/* Budget cards grid */}
      {isLoading ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <Card key={i}>
              <Skeleton className="h-4 w-32 mb-3" />
              <Skeleton className="h-2 w-full mb-3" />
              <Skeleton className="h-3 w-24" />
            </Card>
          ))}
        </div>
      ) : displayBudgets && displayBudgets.length > 0 ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {displayBudgets.map((b) => (
            <BudgetCard key={b.id} budget={b} />
          ))}
        </div>
      ) : (
        <Card hover={false}>
          <div className="py-8 text-center text-[var(--text-muted)]">
            <p className="text-lg mb-1">
              No budgets for {formatMonth(budgetMonth)}
              {filters.category && ` in ${filters.category}`}
            </p>
            <p className="text-sm">Use the form above to create your first budget.</p>
          </div>
        </Card>
      )}
    </div>
  );
}
