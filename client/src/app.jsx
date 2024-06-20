import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Chat } from "./pages/chat/chat";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ChatProvider } from "./context/chat-provider";
import { ThemeProvider } from "./context/theme-provider";
import { LoadingScreen } from "./components/ui/loading-screen";
import { useUserStore } from "./store/user";
import { useEffect } from "react";
import { ErrorPage } from "./components/error-page"
const queryClient = new QueryClient();

export function App() {
  const { isAuthorizing, isAuthorized } = useUserStore();
  useEffect(() => {
    useUserStore.getState().authorize();
  }, []);
  return (
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        {isAuthorizing && <LoadingScreen message="Authorizing your account" />}
        {isAuthorized && (
          <BrowserRouter>
            <Routes>
              <Route element={<ChatProvider />}>
                <Route path="/" element={<Chat />} />
                <Route path="/chat/:chatId" element={<Chat />} />
              </Route>
            </Routes>
          </BrowserRouter>
        )}
        <ErrorPage />
      </QueryClientProvider>
    </ThemeProvider>
  );
}
