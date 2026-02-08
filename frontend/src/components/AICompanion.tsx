import { useState, useRef, useEffect, useCallback } from "react";
import { Mic, MicOff, Volume2 } from "lucide-react";
import {
  companionGreet,
  companionChat,
  type ConversationMessage,
} from "@/lib/api";

interface AICompanionProps {
  sceneContext: string;
  isActive: boolean;
}

type CompanionState =
  | "idle"
  | "greeting"
  | "speaking"
  | "listening"
  | "processing";

export default function AICompanion({ sceneContext, isActive }: AICompanionProps) {
  const [state, setState] = useState<CompanionState>("idle");
  const [conversationHistory, setConversationHistory] = useState<ConversationMessage[]>([]);
  const [currentText, setCurrentText] = useState<string>("");
  const [hasGreeted, setHasGreeted] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const companionAudioRef = useRef<HTMLAudioElement | null>(null);

  // --- Play audio helper ---
  const playAudio = useCallback((audioBase64: string): Promise<void> => {
    return new Promise((resolve, reject) => {
      setState("speaking");

      const binary = atob(audioBase64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: "audio/wav" });
      const url = URL.createObjectURL(blob);

      const audio = new Audio(url);
      companionAudioRef.current = audio;

      audio.onended = () => {
        URL.revokeObjectURL(url);
        companionAudioRef.current = null;
        resolve();
      };
      audio.onerror = () => {
        URL.revokeObjectURL(url);
        companionAudioRef.current = null;
        reject(new Error("Audio playback failed"));
      };

      audio.play().catch(reject);
    });
  }, []);

  // --- Auto-greet when active ---
  useEffect(() => {
    if (!isActive || hasGreeted) return;

    const doGreet = async () => {
      setState("greeting");
      try {
        const result = await companionGreet(sceneContext);
        setHasGreeted(true);
        setConversationHistory([{ role: "assistant", content: result.text }]);
        setCurrentText(result.text);
        await playAudio(result.audio_base64);
      } catch (err) {
        console.error("Companion greeting failed:", err);
      } finally {
        setState("idle");
        setTimeout(() => setCurrentText(""), 5000);
      }
    };

    const timer = setTimeout(doGreet, 2000);
    return () => clearTimeout(timer);
  }, [isActive, hasGreeted, sceneContext, playAudio]);

  // --- Start recording ---
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : "audio/mp4";

      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.start();
      setState("listening");
    } catch (err) {
      console.error("Mic permission denied:", err);
    }
  }, []);

  // --- Stop recording and send ---
  const stopRecording = useCallback(async () => {
    const mediaRecorder = mediaRecorderRef.current;
    if (!mediaRecorder || mediaRecorder.state !== "recording") return;

    return new Promise<void>((resolve) => {
      mediaRecorder.onstop = async () => {
        mediaRecorder.stream.getTracks().forEach((track) => track.stop());

        const audioBlob = new Blob(audioChunksRef.current, {
          type: mediaRecorder.mimeType,
        });
        audioChunksRef.current = [];

        const reader = new FileReader();
        reader.onloadend = async () => {
          const base64 = (reader.result as string).split(",")[1];

          setState("processing");
          setCurrentText("Thinking...");

          try {
            const recentHistory = conversationHistory.slice(-10);
            const result = await companionChat(base64, recentHistory, sceneContext);

            setConversationHistory((prev) => [
              ...prev,
              { role: "user", content: result.user_text },
              { role: "assistant", content: result.text },
            ]);

            setCurrentText(result.text);
            await playAudio(result.audio_base64);
          } catch (err) {
            console.error("Companion chat failed:", err);
            setCurrentText("Sorry, I couldn't hear you clearly...");
          } finally {
            setState("idle");
            setTimeout(() => setCurrentText(""), 5000);
          }

          resolve();
        };
        reader.readAsDataURL(audioBlob);
      };

      mediaRecorder.stop();
    });
  }, [conversationHistory, sceneContext, playAudio]);

  // --- Mic button click ---
  const handleMicClick = useCallback(() => {
    if (state === "listening") {
      stopRecording();
    } else if (state === "idle") {
      startRecording();
    }
  }, [state, startRecording, stopRecording]);

  // --- Cleanup on unmount ---
  useEffect(() => {
    return () => {
      if (companionAudioRef.current) {
        companionAudioRef.current.pause();
        companionAudioRef.current = null;
      }
      if (mediaRecorderRef.current?.state === "recording") {
        mediaRecorderRef.current.stop();
        mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  if (!isActive) return null;

  return (
    <>
      {/* Subtitle overlay */}
      {currentText && (
        <div className="absolute bottom-28 left-1/2 -translate-x-1/2 z-50 max-w-md">
          <div className="glass rounded-2xl px-5 py-3 text-center">
            <p className="text-sm text-foreground/90 leading-relaxed italic">
              {currentText}
            </p>
          </div>
        </div>
      )}

      {/* Mic button - bottom right */}
      <div className="absolute bottom-6 right-6 z-50">
        <button
          onClick={handleMicClick}
          disabled={state === "greeting" || state === "processing" || state === "speaking"}
          className={`
            relative flex h-14 w-14 items-center justify-center rounded-full
            transition-all duration-300
            ${state === "listening"
              ? "bg-red-500/80 text-white scale-110"
              : state === "speaking"
              ? "bg-primary/60 text-primary-foreground"
              : state === "processing" || state === "greeting"
              ? "bg-secondary/60 text-muted-foreground cursor-wait"
              : "glass text-foreground/70 hover:text-foreground hover:scale-105"
            }
            disabled:opacity-50
          `}
        >
          {state === "speaking" ? (
            <Volume2 className="h-6 w-6 animate-pulse" />
          ) : state === "processing" || state === "greeting" ? (
            <div className="h-5 w-5 animate-spin rounded-full border-2 border-current border-t-transparent" />
          ) : state === "listening" ? (
            <MicOff className="h-6 w-6" />
          ) : (
            <Mic className="h-6 w-6" />
          )}

          {/* Pulsing ring when listening */}
          {state === "listening" && (
            <span className="absolute inset-0 rounded-full border-2 border-red-400 animate-ping" />
          )}
        </button>
      </div>
    </>
  );
}
