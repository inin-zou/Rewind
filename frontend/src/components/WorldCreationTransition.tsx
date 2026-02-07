import { useEffect, useState } from "react";
import Hyperspeed from "./Hyperspeed";
import { hyperspeedPresets } from "./HyperSpeedPresets";

interface WorldCreationTransitionProps {
  isActive: boolean;
  onTransitionEnd: () => void;
}

export default function WorldCreationTransition({
  isActive,
  onTransitionEnd,
}: WorldCreationTransitionProps) {
  const [phase, setPhase] = useState<"idle" | "zooming" | "hyperspeed">("idle");

  useEffect(() => {
    if (!isActive) {
      setPhase("idle");
      return;
    }

    // Phase 1: Start zoom-out immediately
    setPhase("zooming");

    // Phase 2: Start Hyperspeed very early (800ms) for immediate feel
    const zoomTimer = setTimeout(() => {
      setPhase("hyperspeed");
    }, 800);

    // Phase 3: After full experience (14s total), trigger redirect
    const redirectTimer = setTimeout(() => {
      onTransitionEnd();
    }, 14000);

    return () => {
      clearTimeout(zoomTimer);
      clearTimeout(redirectTimer);
    };
  }, [isActive, onTransitionEnd]);

  if (!isActive) return null;

  return (
    <>
      {/* Subtle overlay - only during zoom, fades as Hyperspeed starts */}
      <div
        className={`fixed inset-0 z-[100] pointer-events-none transition-opacity duration-[2000ms] ease-in ${
          phase === "zooming"
            ? "bg-background/30 opacity-100"
            : "bg-background/0 opacity-0"
        }`}
      />

      {/* Hyperspeed state - fades in very quickly for immediate feel */}
      {phase === "hyperspeed" && (
        <div className="fixed inset-0 z-[101] animate-in fade-in duration-300">
          <Hyperspeed effectOptions={hyperspeedPresets.one} />
        </div>
      )}
    </>
  );
}
