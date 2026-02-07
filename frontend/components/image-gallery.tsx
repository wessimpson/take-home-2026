"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import Image from "next/image";
import { useCallback, useEffect, useMemo, useState } from "react";

export function ImageGallery({
  images,
  alt,
  videoUrl,
  controlledIndex,
  onIndexChange,
}: {
  images: string[];
  alt: string;
  videoUrl?: string | null;
  /** When provided, the parent controls which image is shown. */
  controlledIndex?: number;
  /** Called whenever the user selects a different image. */
  onIndexChange?: (index: number) => void;
}) {
  const [internalIndex, setInternalIndex] = useState(0);
  const [failedUrls, setFailedUrls] = useState<Set<string>>(() => new Set());

  const isControlled = controlledIndex !== undefined;
  const selectedIndex = isControlled ? controlledIndex : internalIndex;

  // Sync internal state when parent changes controlled index
  useEffect(() => {
    if (isControlled) setInternalIndex(controlledIndex);
  }, [isControlled, controlledIndex]);

  const handleError = useCallback((url: string) => {
    setFailedUrls((prev) => new Set(prev).add(url));
  }, []);

  // Track both the filtered images and their original indices
  const { validImages, originalIndices } = useMemo(() => {
    const valid: string[] = [];
    const indices: number[] = [];
    images.forEach((url, i) => {
      if (!failedUrls.has(url)) {
        valid.push(url);
        indices.push(i);
      }
    });
    return { validImages: valid, originalIndices: indices };
  }, [images, failedUrls]);

  const handleSelect = useCallback(
    (validIdx: number) => {
      const origIdx = originalIndices[validIdx] ?? validIdx;
      if (!isControlled) setInternalIndex(origIdx);
      onIndexChange?.(origIdx);
    },
    [isControlled, onIndexChange, originalIndices],
  );

  // Map selected index (in original space) to valid space
  const clampedIndex = useMemo(() => {
    const validIdx = originalIndices.indexOf(selectedIndex);
    if (validIdx >= 0) return validIdx;
    return Math.min(selectedIndex, Math.max(0, validImages.length - 1));
  }, [selectedIndex, originalIndices, validImages.length]);

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
      <div className="group relative aspect-square overflow-hidden rounded-lg bg-muted">
        <Image
          src={validImages[clampedIndex]}
          alt={alt}
          fill
          sizes="(max-width: 768px) 100vw, 50vw"
          className="object-contain"
          priority
          onError={() => handleError(validImages[clampedIndex])}
        />

        {/* Navigation arrows */}
        {validImages.length > 1 && (
          <>
            <button
              onClick={() => handleSelect((clampedIndex - 1 + validImages.length) % validImages.length)}
              className="absolute left-2 top-1/2 -translate-y-1/2 rounded-full bg-background/80 p-2 shadow-md backdrop-blur-sm transition-opacity hover:bg-background md:opacity-0 md:group-hover:opacity-100"
              aria-label="Previous image"
            >
              <ChevronLeft className="h-5 w-5" />
            </button>
            <button
              onClick={() => handleSelect((clampedIndex + 1) % validImages.length)}
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full bg-background/80 p-2 shadow-md backdrop-blur-sm transition-opacity hover:bg-background md:opacity-0 md:group-hover:opacity-100"
              aria-label="Next image"
            >
              <ChevronRight className="h-5 w-5" />
            </button>
          </>
        )}
      </div>

      {/* Thumbnail strip */}
      {validImages.length > 1 && (
        <div className="flex gap-2 overflow-x-auto pb-1">
          {validImages.map((url, i) => (
            <button
              key={url}
              onClick={() => handleSelect(i)}
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
