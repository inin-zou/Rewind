import { useEffect, useRef, useCallback, useState } from "react";
import { useNavigate } from "react-router-dom";
import InfiniteMenu from "@/components/InfiniteMenu/InfiniteMenu";
import WorldTransition from "@/components/WorldTransition";
import { worlds } from "@/data/worlds";

const menuItems = worlds.map((world) => ({
  image: world.image,
  link: world.link,
  title: world.title,
  description: world.description,
}));

export default function WorldMenu() {
  const navigate = useNavigate();
  const containerRef = useRef<HTMLDivElement>(null);
  const [transitioning, setTransitioning] = useState(false);
  const [activeWorld, setActiveWorld] = useState<(typeof worlds)[0] | null>(
    null
  );
  const [menuMoving, setMenuMoving] = useState(true);

  // Resolve the currently active world from the DOM (title rendered by InfiniteMenu)
  const getActiveWorld = useCallback(() => {
    const container = containerRef.current;
    if (!container) return null;

    const titleEl = container.querySelector(".face-title.active");
    if (!titleEl) return null;

    const title = titleEl.textContent?.trim();
    return worlds.find((w) => w.title === title) ?? null;
  }, []);

  const isMenuMoving = useCallback(() => {
    const container = containerRef.current;
    if (!container) return true;
    // If the title has class "inactive", the menu is moving
    const titleEl = container.querySelector(".face-title");
    if (!titleEl) return true;
    return titleEl.classList.contains("inactive");
  }, []);

  // Poll the active world and movement state
  useEffect(() => {
    let raf: number;
    const poll = () => {
      const world = getActiveWorld();
      const moving = isMenuMoving();
      setActiveWorld((prev) => {
        if (prev?.id !== world?.id) return world;
        return prev;
      });
      setMenuMoving(moving);
      raf = requestAnimationFrame(poll);
    };
    raf = requestAnimationFrame(poll);
    return () => cancelAnimationFrame(raf);
  }, [getActiveWorld, isMenuMoving]);

  // Enter key handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Enter" || transitioning) return;
      if (isMenuMoving()) return;

      const world = getActiveWorld();
      if (!world) return;

      e.preventDefault();
      setActiveWorld(world);
      setTransitioning(true);
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [transitioning, getActiveWorld, isMenuMoving]);

  const handleTransitionEnd = useCallback(() => {
    if (activeWorld) {
      navigate(activeWorld.link);
    }
  }, [activeWorld, navigate]);

  // Keep the old click-based navigation on the action button
  const handleClick = useCallback(
    (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const actionBtn = target.closest(".action-button");
      if (actionBtn) {
        e.preventDefault();
        e.stopPropagation();

        const world = getActiveWorld();
        if (world) {
          setActiveWorld(world);
          setTransitioning(true);
          return;
        }
        navigate("/demo");
      }
    },
    [navigate, getActiveWorld]
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    container.addEventListener("click", handleClick, true);
    return () => container.removeEventListener("click", handleClick, true);
  }, [handleClick]);

  // Determine if hint should show
  const showHint = activeWorld && !transitioning && !menuMoving;

  return (
    <div ref={containerRef} className="w-full h-full">
      <InfiniteMenu items={menuItems} />

      {/* Enter hint */}
      <div
        className={`pointer-events-none absolute bottom-32 left-1/2 -translate-x-1/2 z-30 flex items-center gap-2 transition-all duration-500 ${
          showHint
            ? "opacity-100 translate-y-0"
            : "opacity-0 translate-y-4"
        }`}
      >
        <kbd className="glass-strong rounded-lg px-3 py-1.5 text-xs font-semibold text-foreground tracking-wider uppercase">
          Enter ↵
        </kbd>
        <span className="text-sm text-muted-foreground">to explore</span>
      </div>

      {/* Cinematic transition overlay */}
      <WorldTransition
        isActive={transitioning}
        worldImage={activeWorld?.image ?? ""}
        worldTitle={activeWorld?.title ?? ""}
        onTransitionEnd={handleTransitionEnd}
      />
    </div>
  );
}
