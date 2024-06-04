import { getApiUrl } from "@/lib/get-api-url";
import axios from "axios";
export async function getChats() {
  const response = await axios.get(getApiUrl("/conversations"));
  return response.data;
}
