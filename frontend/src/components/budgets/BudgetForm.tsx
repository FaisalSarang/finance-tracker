"use client";

import { useState } from "react";
import { Plus } from "lucide-react";
import { ALL_CATEGORIES } from "@/lib/constants";
import { createBudget } from "@/lib/api";

interface BudgetFormProps {
  month: string;
  onCreated: () => void;
}

export default function BudgetForm({ month, onCreated }: BudgetFormProps) {
  const [category, setCategory] = useState(ALL_CATEGORIES[0]);
  const [limit, setLimit] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!limit || Number(limit) <= 0) return;
    setLoading(true);
    try {
      await createBudget({ category, monthly_limit: Number(limit), month });
      setLimit("");
      onCreated();
    } catch (err) {
      console.error("Failed to create budget:", err);
    } finally {
      setLoading(false);
    }
  };

  const selectClass =
    "rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] px-3 py-2 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--accent-green)] transition-colors";

  return (
    <form onSubmit={handleSubmit} className="flex flex-wrap items-end gap-3">
      <div>
        <label className="block text-xs text-[var(--text-muted)] mb-1">Category</label>
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className={selectClass}
        >
          {ALL_CATEGORIES.filter((c) => c !== "Income").map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </div>

      <div>
        <label className="block text-xs text-[var(--text-muted)] mb-1">Monthly Limit ($)</label>
        <input
          type="number"
          step="0.01"
          min="0"
          value={limit}
          onChange={(e) => setLimit(e.target.value)}
          placeholder="500.00"
          className={`${selectClass} w-32`}
        />
      </div>

      <button
        type="submit"
        disabled={loading || !limit}
        className="flex items-center gap-2 rounded-lg bg-[var(--accent-green)] px-4 py-2 text-sm font-medium text-[#0B0E13] hover:opacity-90 disabled:opacity-40 transition-opacity"
      >
        <Plus size={16} />
        {loading ? "Creating..." : "Add Budget"}
      </button>
    </form>
  );
}
