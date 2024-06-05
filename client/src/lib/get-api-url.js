export function getApiUrl(uri) {
  let baseUrl = import.meta.env.VITE_API_URL || "http://localhost:5001/";

  return `${baseUrl}${uri.replace(/^\//, "")}`;
}
