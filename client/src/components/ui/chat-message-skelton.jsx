import { cn } from "@/lib/utils";
import { LoadingText } from "./loading-text";
import { Skeleton } from "./skeleton";

export function ChatMessageSkeleton({ className }) {
  return (
    <div className={cn("w-full flex", className)}>
      <Skeleton className="w-10 h-10 shrink-0 mr-4 rounded-full" />
      <LoadingText />
    </div>
  );
}
