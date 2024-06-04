import { getApiUrl } from "@/lib/get-api-url";

export async function createUser() {
  const response = await fetch(getApiUrl(`/user`), {
    method: "POST",
  });
  return await response.json();
}
