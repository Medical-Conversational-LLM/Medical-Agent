export function getApiUrl(uri) {
  let baseUrl = "http://localhost:5001/";

  return `${baseUrl}${uri.replace(/^\//, "")}`;
}
