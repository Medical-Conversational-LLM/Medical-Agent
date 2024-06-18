import { cn } from "@/lib/utils";

import { Bot as BotIcon } from "lucide-react";
export function ChatSuggestions({
  suggestions,
  className,
  innerClassName,
  onSuggestionSelect,
}) {
  return (
    <div className={cn(className)}>
      <div
        className={cn(
          "pt-14 flex h-full justify-between  flex-col",
          innerClassName
        )}
      >
        <div className="flex max-sm:flex-col gap-4 items-center">
          <div className="bg-gradient-to-r from-cyan-500 to-blue-500 text-white rounded-3xl p-4 w-fit">
            <BotIcon size={70} />
          </div>

          <div>
            <div className="text-4xl lg:text-6xl font-black text-primary">
              Hallo !!
            </div>
            <div className="text-2xl lg:text-4xl text-muted-foreground font-light">
              Medical Search Engine
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 w-full">
          {suggestions.map((suggestion, index) => (
            <button
              onClick={() => onSuggestionSelect(suggestion.prompt)}
              key={index}
              className="border p-4 rounded-lg text-start bg-background hover:bg-background/10"
            >
              <div className="font-medium">{suggestion.label}</div>
              <p className="text-muted-foreground">{suggestion.description}</p>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
