"use client";

import useSWR from "swr";
import { getStats, getCategoryBreakdown, getTransactions, getBudgets } from "@/lib/api";
import { useFilters } from "@/lib/FilterContext";
import StatCards from "@/components/dashboard/StatCards";
import SpendingChart from "@/components/dashboard/SpendingChart";
import RecentTransactions from "@/components/dashboard/RecentTransactions";
import BudgetProgress from "@/components/dashboard/BudgetProgress";

export default function DashboardPage() {
  const { filters, monthStart, monthEnd, budgetMonth } = useFilters();

  const apiFilters = {
    category: filters.category || undefined,
    start_date: monthStart,
    end_date: monthEnd,
  };

  const { data: stats, isLoading: statsLoading } = useSWR(
    ["stats", filters],
    () => getStats(apiFilters)
  );
  const { data: categories, isLoading: catLoading } = useSWR(
    ["categories", filters],
    () => getCategoryBreakdown(apiFilters)
  );
  const { data: transactions, isLoading: txnLoading } = useSWR(
    ["recent-txns", filters],
    () => getTransactions({ ...apiFilters, limit: 7 })
  );
  const { data: budgets, isLoading: budgetLoading } = useSWR(
    ["budgets-dash", budgetMonth],
    () => getBudgets(budgetMonth)
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-[var(--text-primary)]">Dashboard</h1>
        <p className="text-sm text-[var(--text-secondary)] mt-1">
          Your financial overview at a glance
        </p>
      </div>

      <StatCards stats={stats} loading={statsLoading} />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <SpendingChart data={categories} loading={catLoading} />
        </div>
        <div>
          <BudgetProgress budgets={budgets} loading={budgetLoading} />
        </div>
      </div>

      <RecentTransactions transactions={transactions} loading={txnLoading} />
    </div>
  );
}
