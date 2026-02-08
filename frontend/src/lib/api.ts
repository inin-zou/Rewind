const API_BASE = import.meta.env.VITE_API_URL || "";

export interface GenerateResponse {
  video_base64: string;
  prompt: string;
  num_frames: number;
  pose: string;
  generation_time_seconds: number;
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

/**
 * Convert a File to base64, ensuring landscape orientation (16:9).
 * Draws the image onto a canvas to strip EXIF rotation and
 * rotates portrait images to landscape.
 */
export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      let { width, height } = img;
      let rotate = false;

      // If portrait, rotate to landscape
      if (height > width) {
        rotate = true;
      }

      const canvas = document.createElement("canvas");
      if (rotate) {
        canvas.width = height;
        canvas.height = width;
      } else {
        canvas.width = width;
        canvas.height = height;
      }

      const ctx = canvas.getContext("2d")!;

      if (rotate) {
        // Rotate 90 degrees clockwise
        ctx.translate(canvas.width, 0);
        ctx.rotate(Math.PI / 2);
      }

      ctx.drawImage(img, 0, 0);

      // Export as JPEG (smaller than PNG for photos)
      const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
      const base64 = dataUrl.split(",")[1];
      resolve(base64);
    };
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}
