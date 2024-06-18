import { ChatMessagesList } from "@/components/chat/chat-messages-list";
import { Layout } from "./layout";
import { ChatInput } from "@/components/chat/chat-input";
import { useEffect, useRef } from "react";
import { ChatSuggestions } from "@/components/chat/chat-suggestions";
import { useChats } from "@/hooks/use-chats";
import { useGlobalChat } from "@/context/chat-provider";
import { ChatMessageSkeleton } from "@/components/ui/chat-message-skelton";
import { Customize } from "@/components/chat/customize";

const suggestions = [
  {
    prompt:
      "How does insulin regulate blood glucose levels?",
    label: "Diabetes Management",
    description: "Insulin’s role in regulating blood sugar levels and metabolism.",
  },
  {
    prompt:
      "What are the typical symptoms of asthma, and how do they manifest?",
    label: "Asthma Symptoms",
    description: "Wheezing, breathlessness, cough, reversible obstruction.",
  },
  {
    prompt:
      "What are the current treatment options for rheumatoid arthritis?",
    label: "Rheumatoid Arthritis Treatment",
    description: "Managing autoimmune joint inflammation and pain.",
  },
  {
    prompt:
      "How do viral and bacterial meningitis differ in terms of presentation and management?",
    label: "Meningitis Types",
    description: "Differentiating viral and bacterial causes, treatments vary.",
  },
];
export function Chat() {
  const inputRef = useRef();
  const listRef = useRef();

  const chats = useChats();
  const chat = useGlobalChat();

  console.log("chat", chat)
  useEffect(() => {
    requestAnimationFrame(() => {
      if (!listRef.current || chat.isLoading) {
        return;
      }
      listRef.current.scrollToEnd();
    });
  }, [chat.data?.messages, chat.isLoading]);

  useEffect(() => {
    requestAnimationFrame(() => {
      if (!listRef.current || chat.isLoading) {
        return;
      }
      listRef.current.scrollToEnd();
    });
  }, [chat.isLoading]);

  useEffect(() => {
    setTimeout(() => {
      inputRef.current.focus();
    }, 100);
  }, [chat.isSending, chat.chatId]);

  return (
    <Layout chats={chats}>
      <div className="flex w-full flex-1 h-full">
        <div className="flex-1 h-full w-full flex items-center flex-col overflow-hidden pb-4">
          <div className="flex-1 w-full overflow-hidden">
            {chat.isLoading && (
              <ChatMessageSkeleton className="max-w-2xl mx-auto p-8 px-4" />
            )}
            {chat?.data?.messages?.length > 0 && (
              <ChatMessagesList
                messages={chat.data.messages}
                className={"flex-1 w-full h-full p-4"}
                innerClassName="max-w-2xl mx-auto w-full"
                ref={listRef}
                isReceiving={chat.isSending}
                lastMessageState={chat.lastMessageState}
              />
            )}
            {!chat.isLoading && !chat?.data?.messages?.length && (
              <ChatSuggestions
                className={"flex-1 w-full p-4 h-full"}
                innerClassName="max-w-2xl mx-auto w-full"
                suggestions={suggestions}
                onSuggestionSelect={chat.setMessage}
              />
            )}
          </div>
          <div className="w-full px-4">
            <ChatInput
              inputRef={inputRef}
              value={chat.message}
              onValueChange={chat.setMessage}
              loading={chat.isSending}
              disabled={chat.isSending || chat.isLoading}
              className={"shrink-0 m-4 max-w-3xl mx-auto"}
              onSubmit={chat.onSubmit}
            />
          </div>
        </div>
        {
          /*
<Customize settings={chat.settings} onUpdate={chat.updateSetting} />
          */
        }
      </div>
    </Layout>
  );
}
