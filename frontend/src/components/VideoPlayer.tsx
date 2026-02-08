import { useEffect, useMemo, useRef, forwardRef, useImperativeHandle } from "react";

export interface VideoPlayerHandle {
  /** Capture the current frame as a base64 JPEG string (no prefix). */
  captureFrame: () => string | null;
}

interface VideoPlayerProps {
  videoBase64: string;
  className?: string;
}

const VideoPlayer = forwardRef<VideoPlayerHandle, VideoPlayerProps>(
  ({ videoBase64, className = "" }, ref) => {
    const videoRef = useRef<HTMLVideoElement>(null);

    const videoSrc = useMemo(() => {
      const binary = atob(videoBase64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
      }
      return URL.createObjectURL(new Blob([bytes], { type: "video/mp4" }));
    }, [videoBase64]);

    useEffect(() => {
      return () => URL.revokeObjectURL(videoSrc);
    }, [videoSrc]);

    useEffect(() => {
      videoRef.current?.play().catch(() => {});
    }, [videoSrc]);

    useImperativeHandle(ref, () => ({
      captureFrame: () => {
        const video = videoRef.current;
        if (!video || video.videoWidth === 0) return null;

        // Seek to last frame before capturing
        if (video.duration) {
          video.currentTime = video.duration - 0.01;
        }

        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d")!;
        ctx.drawImage(video, 0, 0);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
        return dataUrl.split(",")[1];
      },
    }));

    return (
      <video
        ref={videoRef}
        src={videoSrc}
        loop
        muted
        playsInline
        className={`absolute inset-0 w-full h-full object-cover ${className}`}
      />
    );
  }
);

VideoPlayer.displayName = "VideoPlayer";
export default VideoPlayer;
