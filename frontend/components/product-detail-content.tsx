"use client";

import { useState } from "react";

import { ImageGallery } from "@/components/image-gallery";
import { Separator } from "@/components/ui/separator";
import { formatPrice, type ProductDetail } from "@/lib/api";

export function ProductDetailContent({
  product,
  initialVariantSlug,
}: {
  product: ProductDetail;
  initialVariantSlug?: string;
}) {
  const initialIndex = initialVariantSlug
    ? Math.max(
        0,
        (product.visual_variants ?? []).findIndex(
          (vv) => vv.slug === initialVariantSlug
        )
      )
    : 0;

  const [selectedVariantIndex, setSelectedVariantIndex] =
    useState(initialIndex);

  const hasVisualVariants = (product.visual_variants?.length ?? 0) >= 2;
  const hasSale = product.price.compare_at_price != null;

  const displayImages = hasVisualVariants
    ? product.visual_variants[selectedVariantIndex]?.image_urls ?? []
    : product.image_urls;

  // Key for ImageGallery — forces state reset when variant changes
  const galleryKey = hasVisualVariants
    ? product.visual_variants[selectedVariantIndex]?.slug ?? "default"
    : "default";

  const handleVariantClick = (index: number) => {
    setSelectedVariantIndex(index);
    const slug = product.visual_variants[index].slug;
    const url = new URL(window.location.href);
    url.searchParams.set("v", slug);
    window.history.replaceState({}, "", url);
  };

  return (
    <div className="grid gap-8 md:grid-cols-2 lg:gap-12">
      {/* Left: Image gallery */}
      <ImageGallery
        key={galleryKey}
        images={displayImages}
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

        {/* Interactive color swatches (when visual variants exist) */}
        {hasVisualVariants && (
          <div>
            <h3 className="mb-2 text-sm font-medium">
              Color:{" "}
              <span className="font-normal text-muted-foreground">
                {product.visual_variants[selectedVariantIndex]?.label}
              </span>
            </h3>
            <div className="flex flex-wrap gap-2">
              {product.visual_variants.map((vv, index) => (
                <button
                  key={vv.slug}
                  onClick={() => handleVariantClick(index)}
                  className={`rounded-full border px-3 py-1 text-xs transition-colors ${
                    index === selectedVariantIndex
                      ? "border-foreground bg-foreground text-background"
                      : "border-border hover:border-foreground"
                  }`}
                >
                  {vv.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Static color pills (no visual variants) */}
        {!hasVisualVariants && product.colors.length > 0 && (
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
  );
}
