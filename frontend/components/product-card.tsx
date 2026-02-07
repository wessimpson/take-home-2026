import Link from "next/link";

import { ProductCardImage } from "@/components/product-card-image";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { type ProductCard as ProductCardType, formatPrice } from "@/lib/api";

export function ProductCard({
  product,
  priority = false,
}: {
  product: ProductCardType;
  priority?: boolean;
}) {
  const hasSale = product.compare_at_price != null;

  return (
    <Link href={`/products/${product.slug}`}>
      <Card className="group overflow-hidden border-0 bg-transparent shadow-none transition-all duration-200 hover:shadow-lg">
        <div className="relative aspect-square overflow-hidden rounded-lg bg-muted">
          {product.thumbnail_url ? (
            <ProductCardImage
              src={product.thumbnail_url}
              alt={product.name}
              priority={priority}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-muted-foreground">
              No image
            </div>
          )}
          {hasSale && (
            <Badge className="absolute top-2 left-2" variant="destructive">
              Sale
            </Badge>
          )}
        </div>
        <CardContent className="px-1 pt-3 pb-0">
          <p className="text-xs tracking-wide text-muted-foreground uppercase">
            {product.brand}
          </p>
          <h3 className="mt-0.5 text-sm leading-snug font-medium line-clamp-2">
            {product.name}
          </h3>
          <div className="mt-1.5 flex items-center gap-2">
            <span className="text-sm font-semibold">
              {formatPrice(product.price, product.currency)}
            </span>
            {hasSale && (
              <span className="text-xs text-muted-foreground line-through">
                {formatPrice(product.compare_at_price!, product.currency)}
              </span>
            )}
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            {product.category}
            {product.colors?.length > 0 && (
              <span>
                {" \u00B7 "}
                {product.colors.slice(0, 3).join(", ")}
                {product.colors.length > 3 &&
                  ` +${product.colors.length - 3}`}
              </span>
            )}
          </p>
        </CardContent>
      </Card>
    </Link>
  );
}
