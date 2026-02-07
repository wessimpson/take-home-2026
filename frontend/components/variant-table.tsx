import { Badge } from "@/components/ui/badge";
import { type Variant, formatPrice } from "@/lib/api";

export function VariantTable({ variants }: { variants: Variant[] }) {
  if (variants.length === 0) return null;

  // Collect all unique attribute keys across variants
  const attrKeys = Array.from(
    new Set(variants.flatMap((v) => Object.keys(v.attributes)))
  );

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-left text-muted-foreground">
            {attrKeys.map((key) => (
              <th key={key} className="pb-2 pr-4 font-medium capitalize">
                {key}
              </th>
            ))}
            {variants.some((v) => v.sku) && (
              <th className="pb-2 pr-4 font-medium">SKU</th>
            )}
            {variants.some((v) => v.price) && (
              <th className="pb-2 pr-4 font-medium">Price</th>
            )}
            <th className="pb-2 font-medium">Status</th>
          </tr>
        </thead>
        <tbody>
          {variants.map((variant, i) => (
            <tr key={i} className="border-b last:border-0">
              {attrKeys.map((key) => (
                <td key={key} className="py-2 pr-4">
                  {variant.attributes[key] || "\u2014"}
                </td>
              ))}
              {variants.some((v) => v.sku) && (
                <td className="py-2 pr-4 font-mono text-xs text-muted-foreground">
                  {variant.sku || "\u2014"}
                </td>
              )}
              {variants.some((v) => v.price) && (
                <td className="py-2 pr-4">
                  {variant.price
                    ? formatPrice(variant.price.price, variant.price.currency)
                    : "\u2014"}
                </td>
              )}
              <td className="py-2">
                <Badge variant={variant.available ? "secondary" : "outline"}>
                  {variant.available ? "In Stock" : "Out of Stock"}
                </Badge>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
