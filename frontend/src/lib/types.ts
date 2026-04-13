/* TypeScript interfaces mirroring backend Pydantic schemas */

export interface Transaction {
  id: number;
  transaction_id: string;
  date: string;
  description: string;
  amount: number;
  transaction_type: "debit" | "credit";
  category: string | null;
  merchant: string | null;
  confidence: number | null;
  created_at: string | null;
}

export interface Stats {
  total_transactions: number;
  total_spent: number;
  total_income: number;
  net_balance: number;
  category_count: number;
  earliest_date: string | null;
  latest_date: string | null;
}

export interface CategoryBreakdown {
  category: string;
  count: number;
  total: number;
  average: number;
  percentage: number;
}

export interface MonthlyTrend {
  month: string;
  total: number;
  count: number;
  category: string | null;
}

export interface CategorizeResult {
  category: string;
  confidence: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface Budget {
  id: number;
  category: string;
  monthly_limit: number;
  month: string;
  spent: number;
  remaining: number;
  percentage: number;
}

export interface UploadResult {
  total_parsed: number;
  categorized: number;
  inserted: number;
  duplicates_skipped: number;
  time_seconds: number;
  categories: CategoryBreakdown[];
}
