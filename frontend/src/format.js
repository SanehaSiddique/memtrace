import { useEffect, useRef, useState } from "react";

export function formatUsd(value, { decimals } = {}) {
  const n = Number(value) || 0;
  const useDecimals = decimals !== undefined ? decimals : n < 1000 ? 2 : 0;
  return n.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: useDecimals,
    maximumFractionDigits: useDecimals,
  });
}

export function formatCompactUsd(value) {
  const n = Number(value) || 0;
  if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
  if (Math.abs(n) >= 1_000) return `$${(n / 1_000).toFixed(1)}K`;
  return formatUsd(n);
}

export function formatNumber(value) {
  return Number(value || 0).toLocaleString("en-US");
}

// Animates from the previous value to a new one whenever `value` changes —
// used for the hero savings figures so updates feel alive without being gimmicky.
export function useAnimatedNumber(value, durationMs = 800) {
  const [display, setDisplay] = useState(value || 0);
  const fromRef = useRef(value || 0);
  const rafRef = useRef(null);

  useEffect(() => {
    const from = fromRef.current;
    const to = Number(value) || 0;
    const start = performance.now();

    cancelAnimationFrame(rafRef.current);

    function tick(now) {
      const progress = Math.min(1, (now - start) / durationMs);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(from + (to - from) * eased);
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        fromRef.current = to;
      }
    }
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return display;
}
