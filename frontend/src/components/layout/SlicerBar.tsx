"use client";

import { Filter, X, Calendar } from "lucide-react";
import { useFilters } from "@/lib/FilterContext";
import { ALL_CATEGORIES } from "@/lib/constants";

const selectClass =
  "rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-1.5 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--accent-green)] transition-colors cursor-pointer";

const MONTHS = [
  { value: "01", label: "January" },
  { value: "02", label: "February" },
  { value: "03", label: "March" },
  { value: "04", label: "April" },
  { value: "05", label: "May" },
  { value: "06", label: "June" },
  { value: "07", label: "July" },
  { value: "08", label: "August" },
  { value: "09", label: "September" },
  { value: "10", label: "October" },
  { value: "11", label: "November" },
  { value: "12", label: "December" },
];

const currentYear = new Date().getFullYear();
const YEARS = Array.from({ length: currentYear - 2023 + 2 }, (_, i) => 2023 + i);

export default function SlicerBar() {
  const { filters, setCategory, setMonth, clearAll, hasActiveFilters } = useFilters();

  // Parse current month filter (could be "" for all time, or "YYYY-MM")
  const selectedYear = filters.month ? filters.month.split("-")[0] : "";
  const selectedMonthNum = filters.month ? filters.month.split("-")[1] : "";

  const handleMonthChange = (m: string) => {
    if (!m) {
      // "All Months" selected — clear date filter
      setMonth("");
    } else if (selectedYear) {
      setMonth(`${selectedYear}-${m}`);
    } else {
      // Month picked but no year yet — default to current year
      setMonth(`${currentYear}-${m}`);
    }
  };

  const handleYearChange = (y: string) => {
    if (!y) {
      // "All Years" selected — clear date filter
      setMonth("");
    } else if (selectedMonthNum) {
      setMonth(`${y}-${selectedMonthNum}`);
    } else {
      // Year picked but no month yet — default to January
      setMonth(`${y}-01`);
    }
  };

  return (
    <div className="glass-card mb-6 px-4 py-3 flex flex-wrap items-center gap-3">
      {/* Label */}
      <div className="flex items-center gap-2 text-[var(--text-muted)] mr-1">
        <Filter size={14} />
        <span className="text-xs font-medium uppercase tracking-wider">Slicers</span>
      </div>

      <div className="h-5 w-px bg-[var(--border)] hidden sm:block" />

      {/* Category */}
      <select
        value={filters.category}
        onChange={(e) => setCategory(e.target.value)}
        className={selectClass}
      >
        <option value="">All Categories</option>
        {ALL_CATEGORIES.map((c) => (
          <option key={c} value={c}>{c}</option>
        ))}
      </select>

      <div className="h-5 w-px bg-[var(--border)] hidden sm:block" />

      {/* Month + Year dropdowns */}
      <div className="flex items-center gap-2">
        <Calendar size={14} className="text-[var(--text-muted)] shrink-0" />
        <select
          value={selectedMonthNum}
          onChange={(e) => handleMonthChange(e.target.value)}
          className={selectClass}
        >
          <option value="">All Months</option>
          {MONTHS.map((m) => (
            <option key={m.value} value={m.value}>{m.label}</option>
          ))}
        </select>
        <select
          value={selectedYear}
          onChange={(e) => handleYearChange(e.target.value)}
          className={selectClass}
        >
          <option value="">All Years</option>
          {YEARS.map((y) => (
            <option key={y} value={String(y)}>{y}</option>
          ))}
        </select>
      </div>

      {/* Clear all filters */}
      {hasActiveFilters && (
        <button
          onClick={clearAll}
          className="flex items-center gap-1 rounded-lg border border-[var(--border)] px-2.5 py-1.5 text-xs text-[var(--text-muted)] hover:border-[var(--accent-red)] hover:text-[var(--accent-red)] transition-colors ml-auto"
        >
          <X size={12} /> Clear filters
        </button>
      )}
    </div>
  );
}
