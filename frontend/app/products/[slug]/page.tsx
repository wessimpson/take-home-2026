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
import { ImageGallery } from "@/components/image-gallery";
import { VariantTable } from "@/components/variant-table";
import { getProduct, formatPrice } from "@/lib/api";

type Params = Promise<{ slug: string }>;

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
}: {
  params: Params;
}) {
  const { slug } = await params;
  const product = await getProduct(slug);
  if (!product) notFound();

  const hasSale = product.price.compare_at_price != null;

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

      {/* Two-column layout */}
      <div className="grid gap-8 md:grid-cols-2 lg:gap-12">
        {/* Left: Image gallery */}
        <ImageGallery
          images={product.image_urls}
          alt={product.name}
          videoUrl={product.video_url}
        />

        {/* Right: Product info */}
        <div className="space-y-6">
          <div>
            <p className="text-sm tracking-wide text-muted-foreground uppercase">
              {product.brand}
            </p>
            <h1 className="mt-1 text-2xl font-bold tracking-tight lg:text-3xl">
              {product.name}
            </h1>
          </div>

          {/* Price */}
          <div className="flex items-baseline gap-3">
            <span className="text-2xl font-semibold">
              {formatPrice(product.price.price, product.price.currency)}
            </span>
            {hasSale && (
              <span className="text-lg text-muted-foreground line-through">
                {formatPrice(
                  product.price.compare_at_price!,
                  product.price.currency
                )}
              </span>
            )}
          </div>

          {/* Colors */}
          {product.colors.length > 0 && (
            <div>
              <h3 className="mb-2 text-sm font-medium">Colors</h3>
              <div className="flex flex-wrap gap-2">
                {product.colors.map((color) => (
                  <span
                    key={color}
                    className="rounded-full border px-3 py-1 text-xs"
                  >
                    {color}
                  </span>
                ))}
              </div>
            </div>
          )}

          <Separator />

          {/* Description */}
          {product.description && (
            <div>
              <h3 className="mb-2 text-sm font-medium">Description</h3>
              <p className="text-sm leading-relaxed text-muted-foreground whitespace-pre-line">
                {product.description}
              </p>
            </div>
          )}

          {/* Key Features */}
          {product.key_features.length > 0 && (
            <div>
              <h3 className="mb-2 text-sm font-medium">Key Features</h3>
              <ul className="space-y-1 text-sm text-muted-foreground">
                {product.key_features.map((feature, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="mt-1.5 h-1 w-1 flex-shrink-0 rounded-full bg-muted-foreground" />
                    {feature}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

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
