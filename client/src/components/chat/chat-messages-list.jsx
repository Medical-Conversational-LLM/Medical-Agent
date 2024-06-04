import { cn } from "@/lib/utils";
import { ScrollArea } from "../ui/scroll-area";

import { forwardRef, useImperativeHandle, useRef } from "react";
import { ChatMessage } from "./chat-message";
export const ChatMessagesList = forwardRef(function ChatMessagesList(
  { messages, className, isReceiving, innerClassName, lastMessageState },
  ref
) {
  const listRef = useRef(null);

  useImperativeHandle(
    ref,
    () => ({
      scrollToEnd() {
        const element = listRef.current.querySelector(
          "[data-radix-scroll-area-viewport]"
        );
        element.scrollTop = element.scrollHeight;
      },
    }),
    []
  );

  const messagesCount = messages.length;
  return (
    <>
      <ScrollArea className={cn(className)} ref={listRef}>
        <div className={cn(innerClassName, "gap-4 flex flex-col")}>
          {messages.map((message, index) => (
            <ChatMessage
              message={message}
              isLast={index === messagesCount - 1}
              isTyping={isReceiving && index === messagesCount - 1}
              key={message.id}
              customState={
                index === messagesCount - 1 ? lastMessageState : undefined
              }
            />
          ))}
        </div>
      </ScrollArea>
    </>
  );
});
