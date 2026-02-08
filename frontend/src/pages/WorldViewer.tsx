import { useParams, useNavigate, useLocation } from "react-router-dom";
import { getWorldById } from "@/data/worlds";
import { useState, useEffect, useRef, useCallback } from "react";
import VideoPlayer from "@/components/VideoPlayer";
import type { VideoPlayerHandle } from "@/components/VideoPlayer";
import {
  getPendingGeneration,
  getResolvedResult,
  clearGeneration,
} from "@/lib/generationStore";
import { generateWorld } from "@/lib/api";
import type { GenerateResponse } from "@/lib/api";

const POSE_MAP: Record<string, string> = {
  w: "w-3",
  s: "s-3",
  a: "a-3",
  d: "d-3",
  arrowup: "w-3",
  arrowdown: "s-3",
  arrowleft: "a-3",
  arrowright: "d-3",
};

const WorldViewer = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const location = useLocation();

  const isGenerated = id === "generated";
  const world = !isGenerated && id ? getWorldById(id) : undefined;

  const [generationResult, setGenerationResult] =
    useState<GenerateResponse | null>(null);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(isGenerated);
  const [isNavigating, setIsNavigating] = useState(false);
  const [activeKey, setActiveKey] = useState<string | null>(null);

  const [isEyeOpening, setIsEyeOpening] = useState(false);
  const [showUI, setShowUI] = useState(false);

  const videoPlayerRef = useRef<VideoPlayerHandle>(null);
  const isGeneratingRef = useRef(false);

  // Load generation result for generated worlds
  useEffect(() => {
    if (!isGenerated) return;

    const resolved = getResolvedResult();
    if (resolved) {
      setGenerationResult(resolved);
      setIsLoading(false);
      return;
    }

    const pending = getPendingGeneration();
    if (!pending) {
      setGenerationError("No generation in progress. Please create a world first.");
      setIsLoading(false);
      return;
    }

    pending
      .then((result) => {
        setGenerationResult(result);
        setIsLoading(false);
      })
      .catch((err) => {
        setGenerationError(err.message || "Generation failed");
        setIsLoading(false);
      });
  }, [isGenerated]);

  // Eye opening animation
  useEffect(() => {
    const fromCreation = location.state?.fromCreation === true;
    setIsEyeOpening(fromCreation);

    if (fromCreation) {
      const uiTimer = setTimeout(() => setShowUI(true), 4200);
      const clearTimer = setTimeout(() => setIsEyeOpening(false), 4500);
      return () => {
        clearTimeout(uiTimer);
        clearTimeout(clearTimer);
      };
    } else {
      setShowUI(true);
    }
  }, [location]);

  // Generate next chunk from last frame + direction
  const generateNextChunk = useCallback(
    async (pose: string) => {
      if (isGeneratingRef.current || !videoPlayerRef.current) return;

      const frameBase64 = videoPlayerRef.current.captureFrame();
      if (!frameBase64) return;

      isGeneratingRef.current = true;
      setIsNavigating(true);

      try {
        const result = await generateWorld(
          frameBase64,
          generationResult?.prompt || "A scene",
          pose
        );
        setGenerationResult(result);
      } catch (err: any) {
        console.error("Chunk generation failed:", err);
      } finally {
        isGeneratingRef.current = false;
        setIsNavigating(false);
      }
    },
    [generationResult?.prompt]
  );

  // WASD + Arrow key controls
  useEffect(() => {
    if (!isGenerated || !generationResult) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      const pose = POSE_MAP[key];
      if (!pose) return;

      e.preventDefault();
      setActiveKey(key);

      if (!isGeneratingRef.current) {
        generateNextChunk(pose);
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if (POSE_MAP[key]) {
        setActiveKey(null);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [isGenerated, generationResult, generateNextChunk]);

  // Not found (static worlds)
  if (!isGenerated && !world) {
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

  // Error state (generated worlds)
  if (isGenerated && generationError) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-background">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-foreground mb-2">
            Generation Failed
          </h1>
          <p className="text-muted-foreground mb-6">{generationError}</p>
          <button
            onClick={() => {
              clearGeneration();
              navigate("/");
            }}
            className="rounded-xl bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground transition-all hover:brightness-110 glow-primary"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  const displayTitle = isGenerated
    ? generationResult?.prompt || "Your Memory"
    : world!.title;

  return (
    <div className="relative h-screen w-screen overflow-hidden">
      {/* World background */}
      {isGenerated && generationResult ? (
        <div
          className={`absolute inset-0 ${isEyeOpening ? "animate-world-focus" : ""}`}
        >
          <VideoPlayer
            ref={videoPlayerRef}
            videoBase64={generationResult.video_base64}
          />
        </div>
      ) : isGenerated && isLoading ? (
        <div className="absolute inset-0 bg-background flex items-center justify-center">
          <div className="text-center">
            <div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            <p className="text-sm text-muted-foreground">
              Generating your world...
            </p>
          </div>
        </div>
      ) : (
        <div
          className={`absolute inset-0 bg-cover bg-center ${isEyeOpening ? "animate-world-focus" : ""}`}
          style={{ backgroundImage: `url(${world!.image})` }}
        />
      )}

      {/* Navigating overlay */}
      {isNavigating && (
        <div className="absolute inset-0 z-[100] flex items-center justify-center pointer-events-none">
          <div className="glass rounded-2xl px-6 py-4 flex items-center gap-3">
            <div className="h-5 w-5 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            <span className="text-sm font-medium text-foreground">
              Exploring...
            </span>
          </div>
        </div>
      )}

      {/* Eye Opening Effect - Upper Eyelid */}
      {isEyeOpening && (
        <div
          className="fixed top-0 left-0 right-0 z-[200] animate-eye-open-upper pointer-events-none"
          style={{
            height: "60vh",
            background:
              "radial-gradient(ellipse 150% 100% at 50% 100%, rgba(8, 8, 12, 0.95) 0%, rgba(8, 8, 12, 0.98) 60%, rgba(8, 8, 12, 1) 100%)",
            filter: "blur(3px)",
            transformOrigin: "top center",
          }}
        />
      )}

      {/* Eye Opening Effect - Lower Eyelid */}
      {isEyeOpening && (
        <div
          className="fixed bottom-0 left-0 right-0 z-[200] animate-eye-open-lower pointer-events-none"
          style={{
            height: "60vh",
            background:
              "radial-gradient(ellipse 150% 100% at 50% 0%, rgba(8, 8, 12, 0.96) 0%, rgba(8, 8, 12, 0.98) 65%, rgba(8, 8, 12, 1) 100%)",
            filter: "blur(2px)",
            transformOrigin: "bottom center",
          }}
        />
      )}

      {/* UI - Only shown after eye opening completes */}
      {showUI && (
        <div className="absolute inset-0 animate-in fade-in duration-1000">
          <div className="absolute inset-0 bg-gradient-to-b from-background/20 via-transparent to-background/40 pointer-events-none" />

          {/* Top bar */}
          <div className="absolute top-0 left-0 right-0 z-50 flex items-center justify-between p-4">
            <button
              onClick={() => {
                clearGeneration();
                navigate("/");
              }}
              className="glass flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium text-foreground transition-all hover:bg-secondary/80"
            >
              ← Back
            </button>
            {isGenerated && (
              <div className="glass rounded-xl px-4 py-2.5">
                <span className="text-xs text-muted-foreground">
                  {generationResult?.generation_time_seconds
                    ? `${generationResult.generation_time_seconds}s`
                    : ""}
                </span>
              </div>
            )}
          </div>

          {/* Center title (only show briefly) */}
          {!isNavigating && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="text-center">
                <h1 className="text-5xl font-bold text-foreground text-glow mb-4 drop-shadow-2xl">
                  {displayTitle}
                </h1>
              </div>
            </div>
          )}

          {/* WASD controls hint - bottom */}
          {isGenerated && generationResult && (
            <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-50">
              <div className="glass rounded-2xl px-6 py-4">
                <div className="flex flex-col items-center gap-1">
                  <div className="flex gap-1">
                    <Key label="W" active={activeKey === "w" || activeKey === "arrowup"} />
                  </div>
                  <div className="flex gap-1">
                    <Key label="A" active={activeKey === "a" || activeKey === "arrowleft"} />
                    <Key label="S" active={activeKey === "s" || activeKey === "arrowdown"} />
                    <Key label="D" active={activeKey === "d" || activeKey === "arrowright"} />
                  </div>
                </div>
                <p className="text-xs text-muted-foreground text-center mt-2">
                  Press to explore
                </p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

function Key({ label, active }: { label: string; active: boolean }) {
  return (
    <div
      className={`w-10 h-10 rounded-lg flex items-center justify-center text-sm font-bold transition-all ${
        active
          ? "bg-primary text-primary-foreground scale-95 glow-primary"
          : "bg-secondary/60 text-foreground/70 border border-border/30"
      }`}
    >
      {label}
    </div>
  );
}

export default WorldViewer;
