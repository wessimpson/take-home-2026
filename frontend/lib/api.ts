const API_URL = process.env.API_URL || "http://localhost:8000";

// ---------------------------------------------------------------------------
// Types (mirror FastAPI response models)
// ---------------------------------------------------------------------------

export interface ProductCard {
  slug: string;
  name: string;
  brand: string;
  price: number;
  currency: string;
  compare_at_price: number | null;
  thumbnail_url: string | null;
  category: string;
  color_count: number;
  variant_count: number;
}

export interface Price {
  price: number;
  currency: string;
  compare_at_price: number | null;
}

export interface Variant {
  attributes: Record<string, string>;
  sku: string | null;
  gtin: string | null;
  price: Price | null;
  image_url: string | null;
  available: boolean;
}

export interface ProductDetail {
  slug: string;
  name: string;
  brand: string;
  price: Price;
  description: string;
  key_features: string[];
  image_urls: string[];
  video_url: string | null;
  category: string;
  category_breadcrumbs: string[];
  colors: string[];
  variants: Variant[];
}

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------

export async function getProducts(): Promise<ProductCard[]> {
  const res = await fetch(`${API_URL}/api/products`, {
    next: { revalidate: 3600 },
  });
  if (!res.ok) throw new Error(`Failed to fetch products: ${res.status}`);
  return res.json();
}

export async function getProduct(slug: string): Promise<ProductDetail | null> {
  const res = await fetch(`${API_URL}/api/products/${slug}`, {
    next: { revalidate: 3600 },
  });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Failed to fetch product: ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

const SYMBOL_TO_CODE: Record<string, string> = {
  "£": "GBP",
  "€": "EUR",
  "¥": "JPY",
  "₹": "INR",
  "kr": "SEK",
  "$": "USD",
};

export function formatPrice(amount: number, currency: string): string {
  const code = SYMBOL_TO_CODE[currency] ?? currency;
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: code,
      minimumFractionDigits: 2,
    }).format(amount);
  } catch {
    return `${currency}${amount.toFixed(2)}`;
  }
}
