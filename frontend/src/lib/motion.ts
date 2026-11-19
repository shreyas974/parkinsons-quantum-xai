import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import Lenis from "lenis";
import { useEffect } from "react";

gsap.registerPlugin(ScrollTrigger);

const EASE = "power2.out";

/**
 * Smooth wheel scrolling, synced with ScrollTrigger. Off for reduced motion and for touch screens,
 * where native scrolling already feels right.
 */
export function useSmoothScroll(reduced: boolean) {
  useEffect(() => {
    if (reduced || window.matchMedia("(pointer: coarse)").matches) return;
    const lenis = new Lenis({ lerp: 0.12 });
    lenis.on("scroll", ScrollTrigger.update);
    const tick = (time: number) => lenis.raf(time * 1000);
    gsap.ticker.add(tick);
    gsap.ticker.lagSmoothing(0);
    // In-page links glide too, stopping below the fixed menu bar.
    const onClick = (e: MouseEvent) => {
      const a = (e.target as HTMLElement).closest<HTMLAnchorElement>('a[href^="#"]');
      const id = a?.getAttribute("href")?.slice(1);
      const target = id ? document.getElementById(id) : null;
      if (!target) return;
      e.preventDefault();
      lenis.scrollTo(target, { offset: -76 });
      history.replaceState(null, "", `#${id}`);
    };
    document.addEventListener("click", onClick);
    return () => {
      document.removeEventListener("click", onClick);
      gsap.ticker.remove(tick);
      lenis.destroy();
    };
  }, [reduced]);
}

/**
 * Elements marked data-reveal rise into place the first time they reach 80% down the screen;
 * data-reveal-stagger does the same for each child, 0.15 s apart.
 */
export function useReveals(reduced: boolean) {
  useEffect(() => {
    if (reduced) return;
    const ctx = gsap.context(() => {
      gsap.utils.toArray<HTMLElement>("[data-reveal]").forEach((el) => {
        gsap.from(el, { autoAlpha: 0, y: 40, duration: 0.7, ease: EASE, scrollTrigger: { trigger: el, start: "top 80%", once: true } });
      });
      gsap.utils.toArray<HTMLElement>("[data-reveal-stagger]").forEach((el) => {
        gsap.from(el.children, {
          autoAlpha: 0, y: 40, duration: 0.7, stagger: 0.15, ease: EASE,
          scrollTrigger: { trigger: el, start: "top 80%", once: true },
        });
      });
      gsap.from(".hero-copy > *", { autoAlpha: 0, y: 30, duration: 0.8, stagger: 0.1, ease: EASE, delay: 0.15 });
    });
    return () => ctx.revert();
  }, [reduced]);
}

/** Cards marked data-tilt lean towards the pointer, at most 6 degrees, on mouse-driven screens only. */
export function useTilt(reduced: boolean) {
  useEffect(() => {
    if (reduced || !window.matchMedia("(hover: hover) and (pointer: fine)").matches) return;
    let current: HTMLElement | null = null;
    const reset = (el: HTMLElement) => {
      el.style.setProperty("--rx", "0deg");
      el.style.setProperty("--ry", "0deg");
    };
    const move = (e: PointerEvent) => {
      const card = (e.target as HTMLElement).closest<HTMLElement>("[data-tilt]");
      if (current && current !== card) reset(current);
      current = card;
      if (!card) return;
      const r = card.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width - 0.5;
      const y = (e.clientY - r.top) / r.height - 0.5;
      card.style.setProperty("--rx", `${(-y * 6).toFixed(2)}deg`);
      card.style.setProperty("--ry", `${(x * 6).toFixed(2)}deg`);
    };
    document.addEventListener("pointermove", move, { passive: true });
    return () => {
      document.removeEventListener("pointermove", move);
      if (current) reset(current);
    };
  }, [reduced]);
}

export { gsap, ScrollTrigger };
