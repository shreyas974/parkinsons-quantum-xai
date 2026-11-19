import { lazy, Suspense, useMemo } from "react";
import { AboutSection } from "./components/AboutSection";
import { Footer } from "./components/Footer";
import { Hero } from "./components/Hero";
import { Nav } from "./components/Nav";
import { QuantumSection } from "./components/QuantumSection";
import { ResearchSection } from "./components/ResearchSection";
import { ResultsSection } from "./components/ResultsSection";
import { SignalsSection } from "./components/SignalsSection";
import { StaticBackdrop } from "./components/StaticBackdrop";
import { ScreeningSection } from "./components/screening/ScreeningSection";
import { hasWebGL, usePrefersReducedMotion } from "./lib/env";
import { useReveals, useSmoothScroll, useTilt } from "./lib/motion";
import { useScrollAnchors } from "./lib/scroll";

// The 3D layer loads after the page text, in its own chunk.
const Experience = lazy(() => import("./three/Experience"));

export default function App() {
  const reduced = usePrefersReducedMotion();
  const webgl = useMemo(hasWebGL, []);
  const light = useMemo(
    () => window.innerWidth < 700 || (navigator.hardwareConcurrency ?? 8) <= 4,
    [],
  );
  useScrollAnchors();
  useSmoothScroll(reduced);
  useReveals(reduced);
  useTilt(reduced);

  return (
    <>
      <div id="scene-labels" className="scene-labels" aria-hidden="true" />
      {webgl ? (
        <Suspense fallback={<StaticBackdrop />}>
          <Experience reduced={reduced} light={light} />
        </Suspense>
      ) : (
        <StaticBackdrop />
      )}
      <a className="skip" href="#screen">Skip to the screening tool</a>
      <Nav />
      <main>
        <Hero />
        <SignalsSection />
        <QuantumSection />
        <ScreeningSection />
        <ResearchSection />
        <ResultsSection />
        <AboutSection />
      </main>
      <Footer />
    </>
  );
}
