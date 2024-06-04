import { getApiUrl } from "@/lib/get-api-url";
import { safeJSONParse } from "@/lib/utils";
import axios from "axios";
export async function postMessage({
  chatId,
  message,
  messageId,
  onMessage,
  signal,
  settings = {},
}) {
  const response = await fetch(getApiUrl(`/conversations`, "post"), {
    method: "post",
    signal,
    headers: {
      "content-type": "application/json",
      Authorization: axios.defaults.headers.common["Authorization"],
    },
    body: JSON.stringify({
      message,
      chatId,
      messageId,
      settings,
    }),
  });

  const stream = response.body;
  const reader = stream.getReader();

  return new Promise((resolve) => {
    readChunk(reader, resolve, onMessage);
  });
}

const readChunk = (reader, onFinish, onMessage) => {
  reader
    .read()
    .then(({ value, done }) => {
      if (done) {
        onFinish();
        console.log("Stream finished");
        return;
      }
      const chunkString = new TextDecoder()
        .decode(value)
        .substring("data: ".length - 1);
      onMessage(safeJSONParse(chunkString, chunkString));
      readChunk(reader, onFinish, onMessage);
    })
    .catch((error) => {
      console.error(error);
    });
};
