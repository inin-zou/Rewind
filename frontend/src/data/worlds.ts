import world1Img from "@/assets/world-1.jpg";
import world2Img from "@/assets/world-2.jpg";
import world3Img from "@/assets/world-3.jpg";
import world4Img from "@/assets/world-4.jpg";
import world5Img from "@/assets/world-5.jpg";

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
    title: "Celestial Drift",
    description: "A floating kingdom above the clouds",
    image: world1Img,
    link: "/world/celestial-drift",
    isDemo: true,
  },
  {
    id: "abyssal-palace",
    title: "Abyssal Palace",
    description: "Crystal halls beneath the deep ocean",
    image: world2Img,
    link: "/world/abyssal-palace",
  },
  {
    id: "golden-shrine",
    title: "Golden Shrine",
    description: "Ancient temples bathed in sunset",
    image: world3Img,
    link: "/world/golden-shrine",
  },
  {
    id: "neon-district",
    title: "Neon District",
    description: "Cyberpunk city that never sleeps",
    image: world4Img,
    link: "/world/neon-district",
  },
  {
    id: "mycelium-grove",
    title: "Mycelium Grove",
    description: "Enchanted forest of glowing fungi",
    image: world5Img,
    link: "/world/mycelium-grove",
  },
];

export const demoWorld = worlds.find((w) => w.isDemo)!;

export function getWorldById(id: string): World | undefined {
  return worlds.find((w) => w.id === id);
}
