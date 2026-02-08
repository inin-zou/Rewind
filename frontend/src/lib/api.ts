const API_BASE = import.meta.env.VITE_API_URL || "";

export interface GenerateResponse {
  video_base64: string;
  prompt: string;
  num_frames: number;
  pose: string;
  generation_time_seconds: number;
}

export interface SoundResponse {
  audio_base64: string;
  sound_prompt: string;
}

export async function generateWorld(
  imageBase64: string,
  prompt: string,
  pose: string = "w-3"
): Promise<GenerateResponse> {
  const response = await fetch(`${API_BASE}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_base64: imageBase64,
      prompt,
      pose,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: "Unknown error" }));
    throw new Error(error.error || `API error: ${response.status}`);
  }

  return response.json();
}

export async function generateSound(
  imageBase64: string
): Promise<SoundResponse> {
  const response = await fetch(`${API_BASE}/api/generate-sound`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_base64: imageBase64 }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: "Unknown error" }));
    throw new Error(error.error || `API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Convert a File to base64.
 * Draws onto a canvas to strip EXIF rotation metadata.
 * Aspect ratio conversion is handled server-side by fal.ai nano-banana.
 */
export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;

      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0);

      const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
      const base64 = dataUrl.split(",")[1];
      resolve(base64);
    };
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}
