
import { Loader } from "lucide-react";
export function CustomMessageState({message}) {
  return (
    <div className="p-2  font-bold text-primary flex items-center  gap-2">
      <Loader className="animate-spin" size={16} />
      {message.message}
    </div>
  );
}
