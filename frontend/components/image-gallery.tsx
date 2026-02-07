"use client";

import Image from "next/image";
import { useCallback, useMemo, useState } from "react";

export function ImageGallery({
  images,
  alt,
  videoUrl,
}: {
  images: string[];
  alt: string;
  videoUrl?: string | null;
}) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [failedUrls, setFailedUrls] = useState<Set<string>>(() => new Set());

  const handleError = useCallback((url: string) => {
    setFailedUrls((prev) => new Set(prev).add(url));
  }, []);

  const validImages = useMemo(
    () => images.filter((url) => !failedUrls.has(url)),
    [images, failedUrls],
  );

  // Clamp selected index to valid range when images get filtered out
  const clampedIndex = Math.min(selectedIndex, Math.max(0, validImages.length - 1));

  if (validImages.length === 0) {
    return (
      <div className="flex aspect-square items-center justify-center rounded-lg bg-muted text-muted-foreground">
        No images available
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Hero image */}
      <div className="relative aspect-square overflow-hidden rounded-lg bg-muted">
        <Image
          src={validImages[clampedIndex]}
          alt={alt}
          fill
          sizes="(max-width: 768px) 100vw, 50vw"
          className="object-contain"
          priority
          onError={() => handleError(validImages[clampedIndex])}
        />
      </div>

      {/* Thumbnail strip */}
      {validImages.length > 1 && (
        <div className="flex gap-2 overflow-x-auto pb-1">
          {validImages.map((url, i) => (
            <button
              key={url}
              onClick={() => setSelectedIndex(i)}
              className={`relative h-16 w-16 flex-shrink-0 overflow-hidden rounded-md border-2 transition-colors ${
                i === clampedIndex
                  ? "border-foreground"
                  : "border-transparent hover:border-muted-foreground/40"
              }`}
            >
              <Image
                src={url}
                alt={`${alt} ${i + 1}`}
                fill
                sizes="64px"
                className="object-cover"
                onError={() => handleError(url)}
              />
            </button>
          ))}
        </div>
      )}

      {/* Video */}
      {videoUrl && (
        <div className="overflow-hidden rounded-lg">
          <video
            src={videoUrl}
            controls
            className="w-full"
            preload="metadata"
          />
        </div>
      )}
    </div>
  );
}
