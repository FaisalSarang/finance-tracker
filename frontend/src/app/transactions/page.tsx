"use client";

import { useState } from "react";
import useSWR from "swr";
import { getTransactions, deleteTransaction } from "@/lib/api";
import { useFilters } from "@/lib/FilterContext";
import Link from "next/link";
import { Search } from "lucide-react";
import Card from "@/components/ui/Card";
import TransactionTable from "@/components/transactions/TransactionTable";
import { SORT_OPTIONS } from "@/lib/constants";

export default function TransactionsPage() {
  const { filters, monthStart, monthEnd } = useFilters();

  // Page-local controls (search, sort, pagination)
  const [sortBy, setSortBy] = useState("date_desc");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const limit = 25;

  const { data, isLoading, mutate } = useSWR(
    ["transactions", filters, sortBy, search, page],
    () =>
      getTransactions({
        category: filters.category || undefined,
        sort_by: sortBy,
        search: search || undefined,
        start_date: monthStart,
        end_date: monthEnd,
        limit,
        offset: page * limit,
      })
  );

  const handleDelete = async (id: number) => {
    if (!confirm("Delete this transaction?")) return;
    try {
      await deleteTransaction(id);
      mutate();
    } catch (err) {
      console.error("Delete failed:", err);
    }
  };

  const selectClass =
    "rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] px-3 py-2 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--accent-green)] transition-colors";

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[var(--text-primary)]">Transactions</h1>
          <p className="text-sm text-[var(--text-secondary)] mt-1">
            View, filter, and manage your transactions
          </p>
        </div>
        <Link
          href="/add"
          className="flex items-center gap-2 rounded-lg bg-[var(--accent-green)] px-4 py-2 text-sm font-medium text-[#0B0E13] hover:opacity-90 transition-opacity"
        >
          + Add
        </Link>
      </div>

      {/* Page-local: search + sort */}
      <Card hover={false}>
        <div className="flex flex-wrap items-center gap-3">
          <div className="relative flex-1 min-w-[200px]">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" />
            <input
              value={search}
              onChange={(e) => { setSearch(e.target.value); setPage(0); }}
              placeholder="Search descriptions..."
              className="w-full rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] pl-10 pr-3 py-2 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] outline-none focus:border-[var(--accent-green)] transition-colors"
            />
          </div>
          <select value={sortBy} onChange={(e) => setSortBy(e.target.value)} className={selectClass}>
            {SORT_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
      </Card>

      <Card hover={false}>
        <TransactionTable
          transactions={data}
          loading={isLoading}
          onDelete={handleDelete}
        />

        {/* Pagination */}
        {data && data.length > 0 && (
          <div className="flex items-center justify-between border-t border-[var(--border)] pt-4 mt-4">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="rounded-lg border border-[var(--border)] px-3 py-1.5 text-sm text-[var(--text-secondary)] disabled:opacity-30 hover:border-[var(--accent-green)] transition-colors"
            >
              Previous
            </button>
            <span className="text-sm text-[var(--text-muted)]">Page {page + 1}</span>
            <button
              onClick={() => setPage((p) => p + 1)}
              disabled={data.length < limit}
              className="rounded-lg border border-[var(--border)] px-3 py-1.5 text-sm text-[var(--text-secondary)] disabled:opacity-30 hover:border-[var(--accent-green)] transition-colors"
            >
              Next
            </button>
          </div>
        )}
      </Card>
    </div>
  );
}
