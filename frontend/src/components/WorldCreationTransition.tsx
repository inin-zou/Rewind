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
  const [isMouseDown, setIsMouseDown] = useState(false);

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

  // Track mouse/touch down state for text animation
  useEffect(() => {
    if (phase !== "hyperspeed") return;

    const handleMouseDown = () => setIsMouseDown(true);
    const handleMouseUp = () => setIsMouseDown(false);
    const handleTouchStart = () => setIsMouseDown(true);
    const handleTouchEnd = () => setIsMouseDown(false);

    window.addEventListener("mousedown", handleMouseDown);
    window.addEventListener("mouseup", handleMouseUp);
    window.addEventListener("touchstart", handleTouchStart);
    window.addEventListener("touchend", handleTouchEnd);

    return () => {
      window.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("touchstart", handleTouchStart);
      window.removeEventListener("touchend", handleTouchEnd);
    };
  }, [phase]);

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
        <>
          <div className="fixed inset-0 z-[101] animate-in fade-in duration-300">
            <Hyperspeed effectOptions={hyperspeedPresets.one} />
          </div>

          {/* Interactive text overlay */}
          <div
            className="fixed inset-0 z-[102] pointer-events-none flex items-center justify-center"
            style={{
              transform: isMouseDown ? 'translateY(-40px)' : 'translateY(0)',
              opacity: isMouseDown ? 0 : 1,
              transition: 'transform 0.4s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
            }}
          >
            <p className="text-xl font-medium text-foreground/90 text-glow">
              Bringing back memories…
            </p>
          </div>
        </>
      )}
    </>
  );
}
