/* Centralized API fetch wrapper for the FastAPI backend */

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api";

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "Unknown error");
    throw new ApiError(text, res.status);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

/* ── Transactions ── */

export interface TransactionFilters {
  category?: string;
  start_date?: string;
  end_date?: string;
  sort_by?: string;
  limit?: number;
  offset?: number;
  search?: string;
}

export function getTransactions(filters: TransactionFilters = {}) {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") params.set(k, String(v));
  });
  const qs = params.toString();
  return request<import("./types").Transaction[]>(`/transactions${qs ? `?${qs}` : ""}`);
}

export function createTransaction(data: Record<string, unknown>) {
  return request<import("./types").Transaction>("/transactions", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export function updateTransaction(id: number, data: Record<string, unknown>) {
  return request<import("./types").Transaction>(`/transactions/${id}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

export function deleteTransaction(id: number) {
  return request<void>(`/transactions/${id}`, { method: "DELETE" });
}

/* ── Stats & Analytics ── */

export function getStats(filters: { category?: string; start_date?: string; end_date?: string } = {}) {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([k, v]) => {
    if (v) params.set(k, v);
  });
  const qs = params.toString();
  return request<import("./types").Stats>(`/stats${qs ? `?${qs}` : ""}`);
}

export function getCategoryBreakdown(filters: { start_date?: string; end_date?: string } = {}) {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([k, v]) => {
    if (v) params.set(k, v);
  });
  const qs = params.toString();
  return request<import("./types").CategoryBreakdown[]>(`/analytics/by-category${qs ? `?${qs}` : ""}`);
}

export function getMonthlyTrends(filters: { start_date?: string; end_date?: string; category?: string } = {}) {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([k, v]) => {
    if (v) params.set(k, v);
  });
  const qs = params.toString();
  return request<import("./types").MonthlyTrend[]>(`/analytics/monthly${qs ? `?${qs}` : ""}`);
}

/* ── ML Categorization ── */

export function categorize(descriptions: string[]) {
  return request<import("./types").CategorizeResult[]>("/categorize", {
    method: "POST",
    body: JSON.stringify({ descriptions }),
  });
}

/* ── Chat ── */

export function chat(message: string, history: import("./types").ChatMessage[]) {
  return request<{ reply: string }>("/chat", {
    method: "POST",
    body: JSON.stringify({ message, history }),
  });
}

/* ── Budgets ── */

export function getBudgets(month: string) {
  return request<import("./types").Budget[]>(`/budgets?month=${month}`);
}

export function createBudget(data: { category: string; monthly_limit: number; month: string }) {
  return request<import("./types").Budget>("/budgets", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export function updateBudget(id: number, monthly_limit: number) {
  return request<import("./types").Budget>(`/budgets/${id}`, {
    method: "PUT",
    body: JSON.stringify({ monthly_limit }),
  });
}

/* ── Upload ── */

export async function uploadCSV(file: File): Promise<import("./types").UploadResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text().catch(() => "Upload failed");
    throw new ApiError(text, res.status);
  }
  return res.json();
}
