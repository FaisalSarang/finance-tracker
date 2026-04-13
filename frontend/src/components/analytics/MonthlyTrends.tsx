"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { formatCurrency } from "@/lib/utils";
import type { MonthlyTrend } from "@/lib/types";

interface MonthlyTrendsProps {
  data: MonthlyTrend[];
}

function CustomTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ value: number }>; label?: string }) {
  if (!active || !payload?.[0]) return null;
  return (
    <div className="custom-tooltip">
      <p className="text-sm font-medium text-[var(--text-primary)]">{label}</p>
      <p className="text-xs text-[var(--accent-green)]">{formatCurrency(payload[0].value)}</p>
    </div>
  );
}

export default function MonthlyTrendsChart({ data }: MonthlyTrendsProps) {
  // Aggregate by month (sum all categories)
  const monthMap = new Map<string, number>();
  for (const d of data) {
    monthMap.set(d.month, (monthMap.get(d.month) ?? 0) + d.total);
  }
  const chartData = Array.from(monthMap.entries())
    .map(([month, total]) => ({ month, total }))
    .sort((a, b) => a.month.localeCompare(b.month));

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-80 text-[var(--text-muted)] text-sm">
        No monthly data
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={350}>
      <AreaChart data={chartData}>
        <defs>
          <linearGradient id="greenGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#7EE8A8" stopOpacity={0.3} />
            <stop offset="100%" stopColor="#7EE8A8" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis
          dataKey="month"
          tick={{ fill: "var(--text-muted)", fontSize: 12 }}
          axisLine={{ stroke: "var(--border)" }}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: "var(--text-muted)", fontSize: 12 }}
          axisLine={{ stroke: "var(--border)" }}
          tickLine={false}
          tickFormatter={(v) => `$${v}`}
        />
        <Tooltip content={<CustomTooltip />} />
        <Area
          type="monotone"
          dataKey="total"
          stroke="#7EE8A8"
          strokeWidth={2}
          fill="url(#greenGrad)"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
