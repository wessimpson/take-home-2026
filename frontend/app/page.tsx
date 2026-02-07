import { ProductCard } from "@/components/product-card";
import { getProducts } from "@/lib/api";

export default async function CatalogPage() {
  const products = await getProducts();

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
      <header className="mb-10">
        <h1 className="text-3xl font-bold tracking-tight">Products</h1>
        <p className="mt-1 text-muted-foreground">
          {products.length} items extracted from product pages
        </p>
      </header>

      <div className="grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 lg:grid-cols-4">
        {products.map((product, i) => (
          <ProductCard
            key={product.slug}
            product={product}
            priority={i < 4}
          />
        ))}
      </div>
    </div>
  );
}
