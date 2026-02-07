import { useEffect, useState } from "react";

interface WorldTransitionProps {
  isActive: boolean;
  worldImage: string;
  worldTitle: string;
  onTransitionEnd: () => void;
}

export default function WorldTransition({
  isActive,
  worldImage,
  worldTitle,
  onTransitionEnd,
}: WorldTransitionProps) {
  const [phase, setPhase] = useState<"idle" | "expand" | "fade">("idle");

  useEffect(() => {
    if (!isActive) {
      setPhase("idle");
      return;
    }

    // Phase 1: expand the circle
    setPhase("expand");

    const fadeTimer = setTimeout(() => {
      setPhase("fade");
    }, 600);

    const navTimer = setTimeout(() => {
      onTransitionEnd();
    }, 1100);

    return () => {
      clearTimeout(fadeTimer);
      clearTimeout(navTimer);
    };
  }, [isActive, onTransitionEnd]);

  if (!isActive && phase === "idle") return null;

  return (
    <div className="fixed inset-0 z-[100] pointer-events-none">
      {/* Expanding circle overlay */}
      <div
        className={`absolute inset-0 flex items-center justify-center transition-all ease-[cubic-bezier(0.16,1,0.3,1)] ${
          phase === "expand" || phase === "fade"
            ? "scale-100 opacity-100"
            : "scale-0 opacity-0"
        }`}
        style={{
          transitionDuration: "700ms",
        }}
      >
        <div
          className="w-[200vmax] h-[200vmax] rounded-full bg-cover bg-center"
          style={{ backgroundImage: `url(${worldImage})` }}
        />
      </div>

      {/* White flash / fade to white */}
      <div
        className={`absolute inset-0 transition-opacity ease-out ${
          phase === "fade" ? "opacity-100" : "opacity-0"
        }`}
        style={{
          transitionDuration: "500ms",
          background: "hsl(var(--background))",
        }}
      />

      {/* Title overlay */}
      <div
        className={`absolute inset-0 flex items-center justify-center transition-all ease-[cubic-bezier(0.16,1,0.3,1)] ${
          phase === "expand"
            ? "opacity-100 scale-100"
            : phase === "fade"
            ? "opacity-0 scale-110"
            : "opacity-0 scale-75"
        }`}
        style={{ transitionDuration: "500ms" }}
      >
        <h1 className="text-5xl md:text-7xl font-bold text-foreground text-glow tracking-tight">
          {worldTitle}
        </h1>
      </div>
    </div>
  );
}
