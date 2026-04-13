"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useDropzone } from "react-dropzone";
import { Upload, Plus, CheckCircle, FileText, AlertCircle } from "lucide-react";
import Card from "@/components/ui/Card";
import Badge from "@/components/ui/Badge";
import { createTransaction, categorize, uploadCSV } from "@/lib/api";
import { ALL_CATEGORIES } from "@/lib/constants";
import { formatCurrency } from "@/lib/utils";
import type { UploadResult } from "@/lib/types";

export default function AddPage() {
  const router = useRouter();

  // ── Manual form state ──
  const [date, setDate] = useState(new Date().toISOString().slice(0, 10));
  const [description, setDescription] = useState("");
  const [amount, setAmount] = useState("");
  const [type, setType] = useState<"debit" | "credit">("debit");
  const [category, setCategory] = useState("");
  const [autoCategory, setAutoCategory] = useState<string | null>(null);
  const [autoConfidence, setAutoConfidence] = useState<number | null>(null);
  const [formLoading, setFormLoading] = useState(false);
  const [formSuccess, setFormSuccess] = useState(false);

  // ── Upload state ──
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadError, setUploadError] = useState("");

  // Auto-categorize on description blur
  const handleDescBlur = async () => {
    if (!description.trim()) return;
    try {
      const results = await categorize([description]);
      if (results.length > 0) {
        setAutoCategory(results[0].category);
        setAutoConfidence(results[0].confidence);
        if (!category) setCategory(results[0].category);
      }
    } catch {
      // Silently fail
    }
  };

  // Manual form submit
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!description || !amount || !date) return;
    setFormLoading(true);
    try {
      await createTransaction({
        date,
        description,
        amount: Number(amount),
        transaction_type: type,
        category: category || undefined,
      });
      setFormSuccess(true);
      setTimeout(() => {
        setFormSuccess(false);
        setDescription("");
        setAmount("");
        setCategory("");
        setAutoCategory(null);
        setAutoConfidence(null);
      }, 2000);
    } catch (err) {
      console.error("Create failed:", err);
    } finally {
      setFormLoading(false);
    }
  };

  // CSV upload
  const onDrop = useCallback(async (files: File[]) => {
    const file = files[0];
    if (!file) return;
    setUploadLoading(true);
    setUploadError("");
    setUploadResult(null);
    try {
      const result = await uploadCSV(file);
      setUploadResult(result);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploadLoading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"] },
    maxFiles: 1,
  });

  const inputClass =
    "w-full rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] px-3 py-2.5 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] outline-none focus:border-[var(--accent-green)] transition-colors";

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-[var(--text-primary)]">Add Transaction</h1>
        <p className="text-sm text-[var(--text-secondary)] mt-1">
          Add a single transaction or upload a CSV file
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* ── Manual Form ── */}
        <Card hover={false}>
          <h2 className="text-lg font-semibold text-[var(--text-primary)] mb-4 flex items-center gap-2">
            <Plus size={18} className="text-[var(--accent-green)]" />
            Manual Entry
          </h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs text-[var(--text-muted)] mb-1">Description</label>
              <input
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                onBlur={handleDescBlur}
                placeholder="e.g. Starbucks Coffee"
                className={inputClass}
                required
              />
              {autoCategory && (
                <div className="mt-2 flex items-center gap-2">
                  <span className="text-xs text-[var(--text-muted)]">AI suggests:</span>
                  <Badge category={autoCategory} />
                  {autoConfidence !== null && (
                    <span className="text-xs text-[var(--text-muted)]">
                      ({(autoConfidence * 100).toFixed(1)}% confidence)
                    </span>
                  )}
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-[var(--text-muted)] mb-1">Amount ($)</label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  value={amount}
                  onChange={(e) => setAmount(e.target.value)}
                  placeholder="0.00"
                  className={inputClass}
                  required
                />
              </div>
              <div>
                <label className="block text-xs text-[var(--text-muted)] mb-1">Date</label>
                <input
                  type="date"
                  value={date}
                  onChange={(e) => setDate(e.target.value)}
                  className={inputClass}
                  required
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-[var(--text-muted)] mb-1">Type</label>
                <select
                  value={type}
                  onChange={(e) => setType(e.target.value as "debit" | "credit")}
                  className={inputClass}
                >
                  <option value="debit">Expense (Debit)</option>
                  <option value="credit">Income (Credit)</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-[var(--text-muted)] mb-1">Category</label>
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  className={inputClass}
                >
                  <option value="">Auto-detect</option>
                  {ALL_CATEGORIES.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>
            </div>

            <button
              type="submit"
              disabled={formLoading}
              className="w-full flex items-center justify-center gap-2 rounded-lg bg-[var(--accent-green)] px-4 py-2.5 text-sm font-medium text-[#0B0E13] hover:opacity-90 disabled:opacity-40 transition-opacity"
            >
              {formSuccess ? (
                <>
                  <CheckCircle size={16} /> Added Successfully!
                </>
              ) : formLoading ? (
                "Adding..."
              ) : (
                <>
                  <Plus size={16} /> Add Transaction
                </>
              )}
            </button>
          </form>
        </Card>

        {/* ── CSV Upload ── */}
        <Card hover={false}>
          <h2 className="text-lg font-semibold text-[var(--text-primary)] mb-4 flex items-center gap-2">
            <Upload size={18} className="text-[var(--accent-blue)]" />
            CSV Upload
          </h2>

          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
              isDragActive
                ? "border-[var(--accent-green)] bg-[rgba(126,232,168,0.05)]"
                : "border-[var(--border)] hover:border-[var(--accent-blue)]"
            }`}
          >
            <input {...getInputProps()} />
            <FileText size={36} className="mx-auto mb-3 text-[var(--text-muted)]" />
            {uploadLoading ? (
              <p className="text-sm text-[var(--text-secondary)]">Processing...</p>
            ) : isDragActive ? (
              <p className="text-sm text-[var(--accent-green)]">Drop your CSV here</p>
            ) : (
              <>
                <p className="text-sm text-[var(--text-secondary)]">
                  Drag & drop a CSV file, or click to browse
                </p>
                <p className="text-xs text-[var(--text-muted)] mt-1">
                  Must include date, description, and amount columns
                </p>
              </>
            )}
          </div>

          {uploadError && (
            <div className="mt-4 flex items-center gap-2 rounded-lg bg-[rgba(248,113,113,0.1)] border border-[rgba(248,113,113,0.2)] p-3">
              <AlertCircle size={16} className="text-[var(--accent-red)] shrink-0" />
              <p className="text-sm text-[var(--accent-red)]">{uploadError}</p>
            </div>
          )}

          {uploadResult && (
            <div className="mt-4 space-y-3">
              <div className="flex items-center gap-2 rounded-lg bg-[rgba(126,232,168,0.1)] border border-[rgba(126,232,168,0.2)] p-3">
                <CheckCircle size={16} className="text-[var(--accent-green)]" />
                <div className="text-sm">
                  <p className="text-[var(--accent-green)] font-medium">Upload Successful!</p>
                  <p className="text-[var(--text-secondary)]">
                    {uploadResult.inserted} inserted, {uploadResult.duplicates_skipped} duplicates skipped
                    in {uploadResult.time_seconds}s
                  </p>
                </div>
              </div>

              {uploadResult.categories.length > 0 && (
                <div>
                  <p className="text-xs text-[var(--text-muted)] mb-2">Categories detected:</p>
                  <div className="flex flex-wrap gap-2">
                    {uploadResult.categories.map((c) => (
                      <div key={c.category} className="flex items-center gap-1">
                        <Badge category={c.category} />
                        <span className="text-xs text-[var(--text-muted)]">({c.count})</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
