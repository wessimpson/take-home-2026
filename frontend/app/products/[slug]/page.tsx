import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Separator } from "@/components/ui/separator";
import { ProductDetailContent } from "@/components/product-detail-content";
import { VariantTable } from "@/components/variant-table";
import { getProduct } from "@/lib/api";

type Params = Promise<{ slug: string }>;
type SearchParams = Promise<{ v?: string }>;

export async function generateMetadata({
  params,
}: {
  params: Params;
}): Promise<Metadata> {
  const { slug } = await params;
  const product = await getProduct(slug);
  if (!product) return { title: "Product Not Found" };
  return {
    title: `${product.name} - ${product.brand}`,
    description: product.description.slice(0, 160),
  };
}

export default async function ProductPage({
  params,
  searchParams,
}: {
  params: Params;
  searchParams: SearchParams;
}) {
  const { slug } = await params;
  const { v } = await searchParams;
  const product = await getProduct(slug);
  if (!product) notFound();

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Back link */}
      <Link
        href="/"
        className="mb-6 inline-flex items-center gap-1 text-sm text-muted-foreground transition-colors hover:text-foreground"
      >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <path
            d="M10 12L6 8L10 4"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        Back to catalog
      </Link>

      {/* Breadcrumbs */}
      {product.category_breadcrumbs.length > 0 && (
        <Breadcrumb className="mb-4">
          <BreadcrumbList>
            {product.category_breadcrumbs.map((crumb, i) => (
              <span key={i} className="contents">
                {i > 0 && <BreadcrumbSeparator />}
                <BreadcrumbItem className="text-xs">{crumb}</BreadcrumbItem>
              </span>
            ))}
          </BreadcrumbList>
        </Breadcrumb>
      )}

      {/* Two-column layout with visual variant support */}
      <ProductDetailContent product={product} initialVariantSlug={v} />

      {/* Variants section */}
      {product.variants.length > 0 && (
        <div className="mt-12">
          <Separator className="mb-8" />
          <h2 className="mb-4 text-lg font-semibold">
            Variants ({product.variants.length})
          </h2>
          <VariantTable variants={product.variants} />
        </div>
      )}
    </div>
  );
}
