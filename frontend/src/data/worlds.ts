import world1Img from "@/assets/world-1.jpg";
import world2Img from "@/assets/world-2.jpg";
import world3Img from "@/assets/world-3.jpg";
import world4Img from "@/assets/world-4.jpg";
import world5Img from "@/assets/world-5.jpg";
import world6Img from "@/assets/world-6.jpg";

export interface World {
  id: string;
  title: string;
  description: string;
  image: string;
  link: string;
  isDemo?: boolean;
}

export const worlds: World[] = [
  {
    id: "celestial-drift",
    title: "Colares, Dec 2024",
    description: "A cherished memory",
    image: world1Img,
    link: "/world/celestial-drift",
    isDemo: true,
  },
  {
    id: "abyssal-palace",
    title: "Cologne, May 2005",
    description: "A cherished memory",
    image: world2Img,
    link: "/world/abyssal-palace",
  },
  {
    id: "golden-shrine",
    title: "Dolomites, Aug 2010",
    description: "A cherished memory",
    image: world3Img,
    link: "/world/golden-shrine",
  },
  {
    id: "neon-district",
    title: "Paris, Feb 2026",
    description: "A cherished memory",
    image: world4Img,
    link: "/world/neon-district",
  },
  {
    id: "mycelium-grove",
    title: "Paris, Feb 2026",
    description: "A cherished memory",
    image: world5Img,
    link: "/world/mycelium-grove",
  },
  {
    id: "memory-six",
    title: "Colares, Dec 2024",
    description: "A cherished memory",
    image: world6Img,
    link: "/world/memory-six",
  },
];

export const demoWorld = worlds.find((w) => w.isDemo)!;

export function getWorldById(id: string): World | undefined {
  return worlds.find((w) => w.id === id);
}
