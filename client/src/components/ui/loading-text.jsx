import { Skeleton } from "./skeleton";

export function LoadingText() {
  return (
    <div className="space-y-2 w-full">
      <Skeleton className="w-full" />
      <Skeleton className="w-3/4" />
      <Skeleton className="w-2/4" />
      <Skeleton className="w-3/4" />
      <Skeleton className="w-1/3" />
    </div>
  );
}
