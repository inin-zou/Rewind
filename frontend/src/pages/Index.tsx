import WorldMenu from "@/components/WorldMenu";
import CreationBar from "@/components/CreationBar";

const Index = () => {
  return (
    <div className="relative h-screen w-screen overflow-hidden bg-background">
      {/* Header overlay */}
      <header className="pointer-events-none absolute top-0 left-0 right-0 z-40 flex items-center justify-between p-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground text-glow">
            Rewind
          </h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            Enter your memories
          </p>
        </div>
      </header>

      {/* InfiniteMenu takes the full viewport */}
      <div className="absolute inset-0">
        <WorldMenu />
      </div>

      {/* Bottom gradient fade for creation bar */}
      <div className="pointer-events-none absolute bottom-0 left-0 right-0 h-48 bg-gradient-to-t from-background via-background/60 to-transparent" />

      {/* Creation bar */}
      <CreationBar />
    </div>
  );
};

export default Index;
