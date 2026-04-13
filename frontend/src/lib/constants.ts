/* Category colors, icons, and static constants */

export const CATEGORY_CONFIG: Record<string, { color: string; bg: string; icon: string }> = {
  "Food & Dining":       { color: "#F97583", bg: "rgba(249,117,131,0.12)", icon: "UtensilsCrossed" },
  "Groceries":           { color: "#7EE8A8", bg: "rgba(126,232,168,0.12)", icon: "ShoppingCart" },
  "Transportation":      { color: "#58A6FF", bg: "rgba(88,166,255,0.12)",  icon: "Car" },
  "Shopping":            { color: "#F692CE", bg: "rgba(246,146,206,0.12)", icon: "ShoppingBag" },
  "Entertainment":       { color: "#BC8CFF", bg: "rgba(188,140,255,0.12)", icon: "Gamepad2" },
  "Health & Pharmacy":   { color: "#56D4DD", bg: "rgba(86,212,221,0.12)",  icon: "Heart" },
  "Utilities":           { color: "#8B95A8", bg: "rgba(139,149,168,0.12)", icon: "Zap" },
  "Income":              { color: "#7EE8A8", bg: "rgba(126,232,168,0.12)", icon: "TrendingUp" },
};

export const CATEGORY_COLORS = Object.values(CATEGORY_CONFIG).map(c => c.color);

export const ALL_CATEGORIES = Object.keys(CATEGORY_CONFIG);

export const SORT_OPTIONS = [
  { value: "date_desc", label: "Newest First" },
  { value: "date_asc", label: "Oldest First" },
  { value: "amount_desc", label: "Highest Amount" },
  { value: "amount_asc", label: "Lowest Amount" },
];

export const NAV_ITEMS = [
  { href: "/",             label: "Dashboard",    icon: "LayoutDashboard" },
  { href: "/transactions", label: "Transactions", icon: "ArrowLeftRight" },
  { href: "/add",          label: "Add",          icon: "PlusCircle" },
  { href: "/analytics",    label: "Analytics",    icon: "PieChart" },
  { href: "/budgets",      label: "Budgets",      icon: "Wallet" },
  { href: "/chat",         label: "AI Chat",      icon: "MessageSquare" },
];
