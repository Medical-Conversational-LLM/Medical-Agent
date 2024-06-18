import { cn } from "@/lib/utils";
import { HumanDate } from "../ui/human-date";
import {
  UserRound as UserRoundIcon,
  Bot as BotIcon,
  Loader as LoaderIcon,
} from "lucide-react";
import { TypeWriter } from "../ui/type-writer";
import { LoadingText } from "../ui/loading-text";
import { CustomMessageState } from "./custom-message-state";

export function ChatMessage({ message, isTyping, isLast, customState }) {
  const isMe = !!message.user_id || message.me
  return (
    <div className="  p-4 flex " key={message.id}>
      <div
        className={cn(
          "w-10 h-10 shrink-0 mr-4 rounded-full  flex items-center justify-center border",

          isMe
            ? "bg-secondary-foreground/5 text-muted-foreground"
            : "bg-green-500 text-white"
        )}
      >
        {isMe && <UserRoundIcon />}
        {!isMe && !isTyping && <BotIcon />}

        {isTyping && <LoaderIcon className="animate-spin" />}
      </div>

      <div className="flex-1">
        <div className="flex justify-between gap-4">
          <div className="font-bold mb-2">{isMe ? "Me" : "Medical Assistant"}</div>
          {!!message.content && (
            <HumanDate
              date={message.created_at}
              className="text-xs text-muted-foreground"
            />
          )}
        </div>
        <div className="relative bg-white shadow-sm border p-4 rounded-r-3xl rounded-b-3xl">
          <div className={cn("pt-1", isMe && "font-medium")}>
            <TypeWriter
              enabled={isLast && isTyping && !!message.content}
              content={message.content}
            />
            {isTyping && !message.content && <LoadingText />}
          </div>
        </div>
        {!!customState && <CustomMessageState message={customState} />}
      </div>
    </div>
  );
}
