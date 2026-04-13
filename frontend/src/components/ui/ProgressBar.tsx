"use client";

interface ProgressBarProps {
  value: number;       /* 0-100+ */
  max?: number;        /* visual cap, default 100 */
  color?: string;
  height?: string;
}

export default function ProgressBar({
  value,
  max = 100,
  color,
  height = "h-2",
}: ProgressBarProps) {
  const pct = Math.min((value / max) * 100, 100);
  const barColor =
    color ?? (value > 90 ? "#F87171" : value > 70 ? "#FBBF24" : "#7EE8A8");

  return (
    <div className={`w-full rounded-full bg-[var(--border)] ${height}`}>
      <div
        className={`${height} rounded-full transition-all duration-700 ease-out`}
        style={{ width: `${pct}%`, backgroundColor: barColor }}
      />
    </div>
  );
}
