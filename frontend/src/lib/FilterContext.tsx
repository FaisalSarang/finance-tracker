"use client";

import { createContext, useContext, useState, useCallback, type ReactNode } from "react";
import { getCurrentMonth } from "./utils";

export interface Filters {
  category: string;       // "" = all categories
  month: string;          // "YYYY-MM" or "" = all time
}

interface FilterContextValue {
  filters: Filters;
  setCategory: (v: string) => void;
  setMonth: (v: string) => void;
  clearAll: () => void;
  /** First day of selected month (undefined if "all time") */
  monthStart: string | undefined;
  /** Last day of selected month (undefined if "all time") */
  monthEnd: string | undefined;
  /** The YYYY-MM to use for budgets (falls back to current month when "all time") */
  budgetMonth: string;
  hasActiveFilters: boolean;
}

const DEFAULT: Filters = { category: "", month: "" };

const Ctx = createContext<FilterContextValue | null>(null);

function getMonthBounds(month: string): { start: string; end: string } | null {
  if (!month) return null;
  const [y, m] = month.split("-").map(Number);
  const start = `${month}-01`;
  const lastDay = new Date(y, m, 0).getDate();
  const end = `${month}-${String(lastDay).padStart(2, "0")}`;
  return { start, end };
}

export function FilterProvider({ children }: { children: ReactNode }) {
  const [filters, setFilters] = useState<Filters>(DEFAULT);

  const setCategory = useCallback((v: string) => setFilters((f) => ({ ...f, category: v })), []);
  const setMonth = useCallback((v: string) => setFilters((f) => ({ ...f, month: v })), []);
  const clearAll = useCallback(() => setFilters(DEFAULT), []);

  const hasActiveFilters = !!(filters.category || filters.month);

  const bounds = filters.month ? getMonthBounds(filters.month) : null;

  return (
    <Ctx.Provider value={{
      filters,
      setCategory,
      setMonth,
      clearAll,
      monthStart: bounds?.start,
      monthEnd: bounds?.end,
      budgetMonth: filters.month || getCurrentMonth(),
      hasActiveFilters,
    }}>
      {children}
    </Ctx.Provider>
  );
}

export function useFilters(): FilterContextValue {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useFilters must be used within FilterProvider");
  return ctx;
}
