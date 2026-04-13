"use client";

import { cn } from "@/lib/utils";

interface CardProps {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
}

export default function Card({ children, className, hover = true }: CardProps) {
  return (
    <div
      className={cn(
        "glass-card p-5 transition-all duration-300",
        hover && "hover:translate-y-[-2px] hover:shadow-lg",
        className
      )}
    >
      {children}
    </div>
  );
}
