"use client";

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from "recharts";
import { CATEGORY_CONFIG } from "@/lib/constants";
import { formatCurrency } from "@/lib/utils";
import type { CategoryBreakdown } from "@/lib/types";

interface CategoryDonutProps {
  data: CategoryBreakdown[];
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: CategoryBreakdown }> }) {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  return (
    <div className="custom-tooltip">
      <p className="text-sm font-medium text-[var(--text-primary)]">{d.category}</p>
      <p className="text-xs text-[var(--text-secondary)]">
        {formatCurrency(d.total)} — {d.percentage}%
      </p>
      <p className="text-xs text-[var(--text-muted)]">
        {d.count} txns, avg {formatCurrency(d.average)}
      </p>
    </div>
  );
}

export default function CategoryDonut({ data }: CategoryDonutProps) {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-80 text-[var(--text-muted)] text-sm">
        No spending data
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={350}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={70}
          outerRadius={120}
          paddingAngle={3}
          dataKey="total"
          nameKey="category"
          label={({ name, percent }: { name?: string; percent?: number }) =>
            `${name ?? ""} (${((percent ?? 0) * 100).toFixed(0)}%)`
          }
          labelLine={{ stroke: "var(--text-muted)", strokeWidth: 1 }}
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
  );
}
