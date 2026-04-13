"use client";

import { CATEGORY_CONFIG } from "@/lib/constants";

interface BadgeProps {
  category: string;
}

export default function Badge({ category }: BadgeProps) {
  const config = CATEGORY_CONFIG[category] ?? {
    color: "#8B95A8",
    bg: "rgba(139,149,168,0.12)",
  };

  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium"
      style={{ color: config.color, backgroundColor: config.bg }}
    >
      <span
        className="h-1.5 w-1.5 rounded-full"
        style={{ backgroundColor: config.color }}
      />
      {category}
    </span>
  );
}
