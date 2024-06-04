import { getApiUrl } from "@/lib/get-api-url";
import axios from "axios";

export async function getChat(chatId) {
  const response = await axios.get(getApiUrl(`/conversations/${chatId}`));
  return response.data;
}
