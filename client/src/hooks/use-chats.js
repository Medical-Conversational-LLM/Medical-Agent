import { getChats } from "@/api/get-chats";
import { useQuery } from "@tanstack/react-query";

export function useChats() {
  const query = useQuery({
    queryKey: ["chats"],
    queryFn: getChats,
  });

  return query;
}
