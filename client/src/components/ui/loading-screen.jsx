import { Loader } from "lucide-react";

export function LoadingScreen({ message = "Screen is loading" }) {
  return (
    <div className="h-screen gap-2 bg-scree flex items-center justify-center flex-col text-2xl font-light">
      <Loader size={32} className="animate-spin" />
      <h2>{message}</h2>
    </div>
  );
}
