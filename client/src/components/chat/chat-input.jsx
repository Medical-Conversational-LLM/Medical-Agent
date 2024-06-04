import { cn } from "@/lib/utils";
import TextareaAutosize from "react-textarea-autosize";
import { Send as SendIcon, Loader as LoaderIcon } from "lucide-react";
import { Button } from "../ui/button";
import { useEffect } from "react";

export function ChatInput({
  className,
  value,
  onValueChange,
  onSubmit,
  inputRef,
  loading,
  disabled,
}) {
  
  return (
    <div className={cn("relative w-full", className)}>
      <TextareaAutosize
        maxRows={4}
        autoFocus
        disabled={loading || disabled}
        placeholder="Chat with me here .."
        className={cn(
          "border   bg-background shadow-sm ring ring-primary/5 focus-within:outline-none w-full rounded-3xl p-3.5 py-3.5 text-base resize-none",
          (loading || disabled) && "opacity-90"
        )}
        value={value}
        onChange={(e) => onValueChange(e.target.value)}
        ref={inputRef}
        onKeyDown={(e) => {
          if (!e.shiftKey && e.code === "Enter") {
            onSubmit();
            e.preventDefault();
          }
        }}
      />

      <div className=" absolute right-2 top-2 flex">
        <Button
          size="icon"
          variant="ghost"
          className="rounded-full hover:bg-primary hover:text-primary-foreground"
          disabled={loading || disabled}
          onClick={() => {
            if (!loading && !disabled) {
              onSubmit();
            }
          }}
        >
          {!loading && <SendIcon size={16} />}
          {loading && <LoaderIcon className="animate-spin" size={16} />}
        </Button>
      </div>
    </div>
  );
}
