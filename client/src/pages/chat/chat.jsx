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
      "Imagine you're crafting a morning ritual geared towards holistic well-being. Your objective is to weave together a tapestry of activities that nurture your body, mind, and soul. Consider incorporating elements like yoga, meditation, healthy breakfast choices, and gratitude journaling. Your task is to create a detailed blueprint for a morning routine that fosters inner harmony and sets a positive tone for the day ahead. Think expansively and prioritize practices that resonate deeply with your sense of wholeness.",
    label: "Holistic Morning Harmony",
    description: "Design a morning ritual for holistic well-being.",
  },
  {
    prompt:
      "Picture yourself architecting a morning regimen dedicated to mental clarity and focus. Your aim is to construct a sequence of activities that sharpen your cognitive faculties and enhance productivity. Think about including strategies such as mindfulness meditation, brain exercises, goal-setting sessions, and healthy breakfast choices. Your objective is to outline a structured morning routine that primes your mind for optimal performance and empowers you to tackle challenges with clarity and resolve.",
    label: "Mindful Morning Mastery",
    description: "Craft a morning routine to boost mental clarity.",
  },
  {
    prompt:
      "Visualize yourself engineering a morning protocol tailored to physical vitality and energy. Your goal is to devise a series of actions that invigorate your body and cultivate resilience. Consider integrating activities like high-intensity workouts, stretching routines, nutritious breakfast options, and hydration practices. Your task is to draft a comprehensive morning plan that fuels your body with vitality and primes you for a day of active engagement and physical well-being.",
    label: "Energizing Morning Vitality",
    description: "Create a morning regimen to enhance physical vitality.",
  },
  {
    prompt:
      "Envision yourself composing a morning ritual centered around emotional balance and inner peace. Your objective is to construct a framework of practices that nurture your emotional well-being and foster resilience in the face of life's challenges. Think about incorporating activities such as gratitude exercises, journaling, deep breathing exercises, and self-affirmations. Your aim is to outline a nurturing morning routine that cultivates emotional equilibrium and empowers you to navigate the day with grace and serenity.",
    label: "Tranquil Morning Serenity",
    description: "Develop a morning routine for emotional balance.",
  },
];
export function Chat() {
  const inputRef = useRef();
  const listRef = useRef();

  const chats = useChats();
  const chat = useGlobalChat();

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
        <Customize settings={chat.settings} onUpdate={chat.updateSetting} />
      </div>
    </Layout>
  );
}
