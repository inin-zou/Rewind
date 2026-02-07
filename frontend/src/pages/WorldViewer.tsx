import { useParams, useNavigate, useLocation } from "react-router-dom";
import { getWorldById } from "@/data/worlds";
import { useState, useEffect } from "react";

const WorldViewer = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const world = id ? getWorldById(id) : undefined;

  // Detect if arriving from creation flow
  const [isEyeOpening, setIsEyeOpening] = useState(false);
  const [showUI, setShowUI] = useState(false);

  useEffect(() => {
    // Check if we're arriving from the creation flow
    const fromCreation = location.state?.fromCreation === true;
    setIsEyeOpening(fromCreation);

    if (fromCreation) {
      // Show UI after eye opening completes (4s)
      const uiTimer = setTimeout(() => {
        setShowUI(true);
      }, 4200);

      // Clear eye opening state
      const clearTimer = setTimeout(() => {
        setIsEyeOpening(false);
      }, 4500);

      return () => {
        clearTimeout(uiTimer);
        clearTimeout(clearTimer);
      };
    } else {
      // If not from creation, show UI immediately
      setShowUI(true);
    }
  }, [location]);

  if (!world) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-background">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-foreground mb-2">
            World not found
          </h1>
          <p className="text-muted-foreground mb-6">
            This memory world doesn't exist yet.
          </p>
          <button
            onClick={() => navigate("/")}
            className="rounded-xl bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground transition-all hover:brightness-110 glow-primary"
          >
            Back to Worlds
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="relative h-screen w-screen overflow-hidden">
      {/* World background - full screen image */}
      <div
        className={`absolute inset-0 bg-cover bg-center ${isEyeOpening ? 'animate-world-focus' : ''}`}
        style={{ backgroundImage: `url(${world.image})` }}
      />

      {/* Eye Opening Effect - Upper Eyelid */}
      {isEyeOpening && (
        <div
          className="fixed top-0 left-0 right-0 z-[200] animate-eye-open-upper pointer-events-none"
          style={{
            height: '60vh',
            background: 'radial-gradient(ellipse 150% 100% at 50% 100%, rgba(8, 8, 12, 0.95) 0%, rgba(8, 8, 12, 0.98) 60%, rgba(8, 8, 12, 1) 100%)',
            filter: 'blur(3px)',
            transformOrigin: 'top center',
          }}
        />
      )}

      {/* Eye Opening Effect - Lower Eyelid */}
      {isEyeOpening && (
        <div
          className="fixed bottom-0 left-0 right-0 z-[200] animate-eye-open-lower pointer-events-none"
          style={{
            height: '60vh',
            background: 'radial-gradient(ellipse 150% 100% at 50% 0%, rgba(8, 8, 12, 0.96) 0%, rgba(8, 8, 12, 0.98) 65%, rgba(8, 8, 12, 1) 100%)',
            filter: 'blur(2px)',
            transformOrigin: 'bottom center',
          }}
        />
      )}

      {/* UI - Only shown after eye opening completes */}
      {showUI && (
        <div className="absolute inset-0 animate-in fade-in duration-1000">
          {/* Subtle overlay for readability */}
          <div className="absolute inset-0 bg-gradient-to-b from-background/20 via-transparent to-background/40" />

          {/* Top bar */}
          <div className="absolute top-0 left-0 right-0 z-50 flex items-center justify-between p-4">
            <button
              onClick={() => navigate("/")}
              className="glass flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium text-foreground transition-all hover:bg-secondary/80"
            >
              ← Back
            </button>
            <div className="glass flex items-center gap-3 rounded-xl px-4 py-2.5">
              <h2 className="text-sm font-semibold text-foreground">
                {world.title}
              </h2>
              <span className="text-xs text-muted-foreground">
                {world.description}
              </span>
            </div>
          </div>

          {/* Center content - minimal */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <h1 className="text-6xl font-bold text-foreground text-glow mb-4 drop-shadow-2xl">
                {world.title}
              </h1>
              <p className="text-xl text-foreground/80 mb-8 drop-shadow-lg">
                {world.description}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default WorldViewer;
