import { useChat } from "@/hooks/use-chat";
import { useParams, Outlet } from "react-router-dom";
import { createContext, useContext } from "react";

const Context = createContext();

export function ChatProvider() {
  const { chatId } = useParams();

  const chat = useChat(chatId);
  
  return (
    <Context.Provider value={{ chat }}>
      <Outlet />
    </Context.Provider>
  );
}

export const useGlobalChat = () => useContext(Context).chat;
