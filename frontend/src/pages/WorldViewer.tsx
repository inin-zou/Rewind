import { useParams, useNavigate } from "react-router-dom";
import { getWorldById } from "@/data/worlds";
import { ArrowLeft, Maximize2 } from "lucide-react";

const WorldViewer = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const world = id ? getWorldById(id) : undefined;

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
    <div className="relative h-screen w-screen overflow-hidden bg-background">
      {/* Top bar */}
      <div className="absolute top-0 left-0 right-0 z-50 flex items-center justify-between p-4">
        <button
          onClick={() => navigate("/")}
          className="glass flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium text-foreground transition-all hover:bg-secondary/80"
        >
          <ArrowLeft className="h-4 w-4" />
          Back
        </button>
        <div className="glass flex items-center gap-3 rounded-xl px-4 py-2.5">
          <h2 className="text-sm font-semibold text-foreground">
            {world.title}
          </h2>
          <span className="text-xs text-muted-foreground">
            {world.description}
          </span>
        </div>
        <button className="glass flex items-center gap-2 rounded-xl px-3 py-2.5 text-sm text-muted-foreground transition-all hover:text-foreground hover:bg-secondary/80">
          <Maximize2 className="h-4 w-4" />
        </button>
      </div>

      {/* 3D World content - placeholder iframe or image */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${world.image})` }}
        />
        <div className="absolute inset-0 bg-background/40 backdrop-blur-sm" />
        <div className="relative z-10 text-center">
          <div className="animate-float">
            <div className="mx-auto mb-6 h-32 w-32 overflow-hidden rounded-full ring-4 ring-primary/30 glow-primary">
              <img
                src={world.image}
                alt={world.title}
                className="h-full w-full object-cover"
              />
            </div>
          </div>
          <h1 className="text-5xl font-bold text-foreground text-glow mb-3">
            {world.title}
          </h1>
          <p className="text-lg text-muted-foreground mb-8">
            {world.description}
          </p>
          <div className="flex items-center justify-center gap-4">
            <button className="rounded-xl bg-primary px-8 py-3 text-sm font-semibold text-primary-foreground transition-all hover:brightness-110 glow-primary">
              Enter World
            </button>
            <button
              onClick={() => navigate("/")}
              className="rounded-xl border border-border px-8 py-3 text-sm font-medium text-muted-foreground transition-all hover:text-foreground hover:border-muted-foreground"
            >
              Go Back
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WorldViewer;
