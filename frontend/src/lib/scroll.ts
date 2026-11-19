import { useEffect } from "react";
import { scrollListeners, scrollState } from "./store";

/** Continuous section index for a scroll position, given each anchor section's top edge. */
export function anchorAt(probe: number, tops: number[]): number {
  if (tops.length === 0 || probe <= tops[0]) return 0;
  for (let i = 0; i < tops.length - 1; i++) {
    if (probe < tops[i + 1]) return i + (probe - tops[i]) / Math.max(1, tops[i + 1] - tops[i]);
  }
  return tops.length - 1;
}

/** How dim the 3D layer gets behind the tool sections: full scene until the qubit story ends. */
export function sceneOpacity(anchor: number): number {
  const dim = Math.min(1, Math.max(0, (anchor - 2.82) / 0.18));
  return 1 - 0.72 * dim;
}

const NAV = 80;

/** Keeps scrollState.anchor in step with the page, dims the 3D scene behind the tool, and drives the progress bar. */
export function useScrollAnchors() {
  useEffect(() => {
    let frame = 0;
    const update = () => {
      frame = 0;
      const sections = Array.from(document.querySelectorAll<HTMLElement>("[data-anchor]"));
      // A pinned section sits inside GSAP's spacer, which keeps its place in the page while the section is fixed.
      const tops = sections.map((s, i) => {
        const el = s.parentElement?.classList.contains("pin-spacer") ? s.parentElement : s;
        return Math.max(i === 0 ? NAV : 0, el.getBoundingClientRect().top + window.scrollY);
      });
      const a = anchorAt(window.scrollY + NAV, tops);
      scrollState.anchor = a;
      const root = document.documentElement;
      root.style.setProperty("--scene-opacity", String(sceneOpacity(a)));
      const max = Math.max(1, root.scrollHeight - window.innerHeight);
      root.style.setProperty("--progress", String(Math.min(1, window.scrollY / max)));
      scrollListeners.forEach((l) => l());
    };
    const onScroll = () => {
      if (!frame) frame = requestAnimationFrame(update);
    };
    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      cancelAnimationFrame(frame);
    };
  }, []);
}
