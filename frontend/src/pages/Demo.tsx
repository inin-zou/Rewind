import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { demoWorld } from "@/data/worlds";

const Demo = () => {
  const navigate = useNavigate();

  useEffect(() => {
    // Redirect to the demo world
    navigate(`/world/${demoWorld.id}`, { replace: true });
  }, [navigate]);

  return (
    <div className="flex h-screen w-screen items-center justify-center bg-background">
      <div className="text-center">
        <div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
        <p className="text-sm text-muted-foreground">
          Loading demo world…
        </p>
      </div>
    </div>
  );
};

export default Demo;
