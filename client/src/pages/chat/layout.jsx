import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  X as CloseIcon,
  MessageSquareText as MessageSquareTextIcon,
  Menu as MenuIcon,
} from "lucide-react";
import { ThemeSwitch } from "@/components/ui/theme-switch";
import { ScrollArea } from "@/components/ui/scroll-area";
import { LLMSelector } from "@/components/chat/llm-selector";

export function Layout({ children, selectedModel, onModelSelect, chats }) {
  const [sideBarOpened, setIsSideBarOpened] = useState(false);

  const { pathname } = useLocation();

  return (
    <div className="flex overflow-hidden h-screen w-screen bg-secondary">
      <div
        className={cn(
          "bg-background",
          "lg:flex flex-col",
          !sideBarOpened && "hidden",
          sideBarOpened && "flex",
          "fixed lg:relative z-50",
          "max-lg:w-[calc(100vw-2rem)] max-lg:left-1/2 max-lg:-translate-x-1/2",
          "max-lg:h-[calc(100vh-2rem)] max-lg:top-1/2 max-lg:-translate-y-1/2",
          "rounded-xl shadow-sm border",
          "lg:w-64 lg:h-full  lg:border-y-none lg:border-l-none lg:border-r lg:shadow-none",
          "lg:transform-none lg:rounded-none"
        )}
      >
        <div className="px-4 mt-4">
          <Link to="/">
            <Button className="w-full">New Chat</Button>
          </Link>
        </div>
        <div className="h-px bg-border mt-4" />
        <ScrollArea className="flex-1 pt-2" viewportClass="p-2">
          {chats?.data?.map?.((item) => (
            <Link
              key={item.id}
              to={`/chat/${item.id}`}
              className={cn(
                "flex items-center truncate gap-3 rounded-lg px-3 py-2 text-secondary-foreground transition-all",
                "hover:bg-secondary/70",
                `/chat/${item.id}` === pathname && "bg-secondary"
              )}
            >
              <MessageSquareTextIcon size={14} className="shrink-0" />
              <div className="w-fit truncate">{item.title}</div>
            </Link>
          ))}
        </ScrollArea>

        <div className="pb-4 px-4">
          <Button
            size="sm"
            variant="outline"
            className="w-full flex gap-2 lg:hidden shrink-0"
            onClick={() => setIsSideBarOpened((status) => !status)}
          >
            <CloseIcon size={18} />
            Close
          </Button>
        </div>
      </div>
      <div className="flex-1 h-full w-full flex items-center flex-col overflow-hidden pt-8 lg:pt-0 bg-background/40">
        {children}
      </div>

      <div className="fixed top-2 right-2 flex gap-4 items-center" id="layout-actions">
        <LLMSelector selected={selectedModel} onSelect={onModelSelect} />
        <Button
          variant="outline"
          className="rounded-full lg:hidden"
          size="icon"
          onClick={() => setIsSideBarOpened((status) => !status)}
        >
          <MenuIcon />
        </Button>
        <ThemeSwitch />
      </div>
    </div>
  );
}
