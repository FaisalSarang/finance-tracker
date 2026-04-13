"use client";

import useSWR from "swr";
import { getCategoryBreakdown, getMonthlyTrends } from "@/lib/api";
import { useFilters } from "@/lib/FilterContext";
import Card from "@/components/ui/Card";
import Skeleton from "@/components/ui/Skeleton";
import CategoryDonut from "@/components/analytics/CategoryDonut";
import MonthlyTrendsChart from "@/components/analytics/MonthlyTrends";
import { formatCurrency } from "@/lib/utils";
import { CATEGORY_CONFIG } from "@/lib/constants";

export default function AnalyticsPage() {
  const { filters, monthStart, monthEnd } = useFilters();

  const apiFilters = {
    start_date: monthStart,
    end_date: monthEnd,
  };

  const { data: categories, isLoading: catLoading } = useSWR(
    ["analytics-cat", filters],
    () => getCategoryBreakdown(apiFilters)
  );

  const { data: monthly, isLoading: monthlyLoading } = useSWR(
    ["analytics-monthly", filters],
    () => getMonthlyTrends({ ...apiFilters, category: filters.category || undefined })
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-[var(--text-primary)]">Analytics</h1>
        <p className="text-sm text-[var(--text-secondary)] mt-1">
          Visualize your spending patterns
          {filters.category && <span className="text-[var(--accent-green)]"> — {filters.category}</span>}
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Category Breakdown Donut */}
        <Card hover={false}>
          <p className="text-sm text-[var(--text-secondary)] mb-4">Spending by Category</p>
          {catLoading ? (
            <Skeleton className="h-80 w-full rounded-xl" />
          ) : (
            <CategoryDonut data={categories ?? []} />
          )}
        </Card>

        {/* Monthly Trends */}
        <Card hover={false}>
          <p className="text-sm text-[var(--text-secondary)] mb-4">Monthly Spending Trend</p>
          {monthlyLoading ? (
            <Skeleton className="h-80 w-full rounded-xl" />
          ) : (
            <MonthlyTrendsChart data={monthly ?? []} />
          )}
        </Card>
      </div>

      {/* Category details table */}
      {categories && categories.length > 0 && (
        <Card hover={false}>
          <p className="text-sm text-[var(--text-secondary)] mb-4">Category Details</p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left py-3 px-3 text-[var(--text-muted)] font-medium">Category</th>
                  <th className="text-right py-3 px-3 text-[var(--text-muted)] font-medium">Transactions</th>
                  <th className="text-right py-3 px-3 text-[var(--text-muted)] font-medium">Total</th>
                  <th className="text-right py-3 px-3 text-[var(--text-muted)] font-medium">Average</th>
                  <th className="text-right py-3 px-3 text-[var(--text-muted)] font-medium">Share</th>
                </tr>
              </thead>
              <tbody>
                {categories.map((cat) => (
                  <tr key={cat.category} className="border-b border-[var(--border)] border-opacity-50">
                    <td className="py-3 px-3">
                      <div className="flex items-center gap-2">
                        <span
                          className="h-2.5 w-2.5 rounded-full"
                          style={{ backgroundColor: CATEGORY_CONFIG[cat.category]?.color ?? "#8B95A8" }}
                        />
                        <span className="text-[var(--text-primary)]">{cat.category}</span>
                      </div>
                    </td>
                    <td className="text-right py-3 px-3 text-[var(--text-secondary)] font-mono-numbers">
                      {cat.count}
                    </td>
                    <td className="text-right py-3 px-3 text-[var(--text-primary)] font-mono-numbers">
                      {formatCurrency(cat.total)}
                    </td>
                    <td className="text-right py-3 px-3 text-[var(--text-secondary)] font-mono-numbers">
                      {formatCurrency(cat.average)}
                    </td>
                    <td className="text-right py-3 px-3 text-[var(--accent-green)] font-mono-numbers">
                      {cat.percentage}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
