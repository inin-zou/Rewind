import { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import WorldMenu from "@/components/WorldMenu";
import CreationBar from "@/components/CreationBar";
import WorldCreationTransition from "@/components/WorldCreationTransition";
import { worlds } from "@/data/worlds";

const Index = () => {
  const navigate = useNavigate();
  const [isCreating, setIsCreating] = useState(false);

  const handleCreateStart = useCallback(() => {
    setIsCreating(true);
  }, []);

  const handleTransitionEnd = useCallback(() => {
    // Pick a random world to redirect to
    const randomWorld = worlds[Math.floor(Math.random() * worlds.length)];
    navigate(randomWorld.link, { state: { fromCreation: true } });
  }, [navigate]);

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-background">
      {/* Main content with zoom-out animation */}
      <div
        className={`absolute inset-0 transition-all duration-[9900ms] ease-out ${
          isCreating
            ? "scale-[0.3] opacity-0"
            : "scale-100 opacity-100"
        }`}
      >
        {/* Header overlay */}
        <header className="pointer-events-none absolute top-0 left-0 right-0 z-40 flex items-center justify-between p-6">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-foreground text-glow">
              Rewind
            </h1>
            <p className="text-xs text-muted-foreground mt-0.5">
              Enter your memories
            </p>
          </div>
        </header>

        {/* InfiniteMenu takes the full viewport */}
        <div className="absolute inset-0">
          <WorldMenu />
        </div>

        {/* Bottom gradient fade for creation bar */}
        <div className="pointer-events-none absolute bottom-0 left-0 right-0 h-48 bg-gradient-to-t from-background via-background/60 to-transparent" />

        {/* Creation bar */}
        <CreationBar onCreateStart={handleCreateStart} />
      </div>

      {/* Creation transition overlay */}
      <WorldCreationTransition
        isActive={isCreating}
        onTransitionEnd={handleTransitionEnd}
      />
    </div>
  );
};

export default Index;
