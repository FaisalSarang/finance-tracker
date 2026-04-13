"use client";

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";
import Card from "@/components/ui/Card";
import Skeleton from "@/components/ui/Skeleton";
import { CATEGORY_CONFIG } from "@/lib/constants";
import { formatCurrency } from "@/lib/utils";
import type { CategoryBreakdown } from "@/lib/types";

interface SpendingChartProps {
  data: CategoryBreakdown[] | undefined;
  loading: boolean;
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: CategoryBreakdown }> }) {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  return (
    <div className="custom-tooltip">
      <p className="text-sm font-medium text-[var(--text-primary)]">{d.category}</p>
      <p className="text-xs text-[var(--text-secondary)]">
        {formatCurrency(d.total)} ({d.percentage}%)
      </p>
      <p className="text-xs text-[var(--text-muted)]">{d.count} transactions</p>
    </div>
  );
}

export default function SpendingChart({ data, loading }: SpendingChartProps) {
  if (loading || !data) {
    return (
      <Card>
        <Skeleton className="h-4 w-40 mb-4" />
        <Skeleton className="h-64 w-full rounded-xl" />
      </Card>
    );
  }

  if (data.length === 0) {
    return (
      <Card>
        <p className="text-sm text-[var(--text-secondary)]">Spending by Category</p>
        <div className="flex items-center justify-center h-64 text-[var(--text-muted)] text-sm">
          No spending data yet
        </div>
      </Card>
    );
  }

  return (
    <Card>
      <p className="text-sm text-[var(--text-secondary)] mb-4">Spending by Category</p>
      <div className="flex flex-col lg:flex-row items-center gap-6">
        <div className="w-full lg:w-1/2 h-64">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={3}
                dataKey="total"
                nameKey="category"
              >
                {data.map((entry) => (
                  <Cell
                    key={entry.category}
                    fill={CATEGORY_CONFIG[entry.category]?.color ?? "#8B95A8"}
                    stroke="transparent"
                  />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="w-full lg:w-1/2 space-y-2">
          {data.map((cat) => {
            const color = CATEGORY_CONFIG[cat.category]?.color ?? "#8B95A8";
            return (
              <div key={cat.category} className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
                  <span className="text-[var(--text-secondary)]">{cat.category}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="font-mono-numbers text-[var(--text-primary)]">
                    {formatCurrency(cat.total)}
                  </span>
                  <span className="text-xs text-[var(--text-muted)] w-12 text-right">
                    {cat.percentage}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </Card>
  );
}
