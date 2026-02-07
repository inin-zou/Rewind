import { useState, useRef, type DragEvent, type ChangeEvent } from "react";
import { Sparkles, Upload, Image as ImageIcon, X } from "lucide-react";

interface CreationBarProps {
  onCreateStart?: () => void;
}

export default function CreationBar({ onCreateStart }: CreationBarProps) {
  const [prompt, setPrompt] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile?.type.startsWith("image/")) {
      setFile(droppedFile);
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile?.type.startsWith("image/")) {
      setFile(selectedFile);
    }
  };

  const handleSubmit = () => {
    if (!prompt.trim() && !file) return;

    setIsGenerating(true);

    // Trigger the parent's zoom-out animation
    if (onCreateStart) {
      onCreateStart();
    }
  };

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 p-4 pb-6">
      <div className="mx-auto max-w-2xl">
        <div
          className={`glass-strong rounded-2xl p-3 transition-all duration-300 ${
            isDragging
              ? "ring-2 ring-primary/50 scale-[1.02]"
              : "ring-1 ring-border/30"
          } ${isGenerating ? "opacity-80" : ""}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {/* File preview */}
          {file && (
            <div className="mb-2 flex items-center gap-2 rounded-lg bg-secondary/50 px-3 py-2 text-sm text-muted-foreground">
              <ImageIcon className="h-4 w-4 text-primary" />
              <span className="flex-1 truncate">{file.name}</span>
              <button
                onClick={() => setFile(null)}
                className="text-muted-foreground hover:text-foreground transition-colors"
                disabled={isGenerating}
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          <div className="flex items-center gap-2">
            {/* Image upload button */}
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
              disabled={isGenerating}
              title="Upload an image"
            >
              <Upload className="h-5 w-5" />
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />

            {/* Text input */}
            <input
              type="text"
              placeholder={
                isGenerating
                  ? "Generating world…"
                  : "Describe your memory world…"
              }
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSubmit();
              }}
              disabled={isGenerating}
              className="flex-1 bg-transparent text-foreground placeholder:text-muted-foreground outline-none text-sm disabled:opacity-50"
            />

            {/* Create button */}
            <button
              onClick={handleSubmit}
              disabled={isGenerating || (!prompt.trim() && !file)}
              className={`flex h-10 shrink-0 items-center gap-2 rounded-xl px-4 text-sm font-semibold transition-all duration-200 ${
                isGenerating
                  ? "bg-primary/50 text-primary-foreground/70 cursor-wait"
                  : "bg-primary text-primary-foreground hover:brightness-110 glow-primary disabled:opacity-30 disabled:cursor-not-allowed disabled:shadow-none"
              }`}
            >
              <Sparkles
                className={`h-4 w-4 ${isGenerating ? "animate-spin" : ""}`}
              />
              {isGenerating ? "Generating…" : "Create"}
            </button>
          </div>
        </div>

        {/* Hint text */}
        <p className="mt-2 text-center text-xs text-muted-foreground/60">
          {isDragging
            ? "Drop your image here"
            : "Type a prompt or drop an image to create a new world"}
        </p>
      </div>
    </div>
  );
}
