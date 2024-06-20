import { getApiUrl } from "@/lib/get-api-url";
import { safeJSONParse } from "@/lib/utils";
import axios from "axios";
export async function postMessage({
  chatId,
  message,
  messageId,
  onMessage,
  model = "self-reflective",
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
      model
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
      let chunkString = new TextDecoder()
        .decode(value)

        chunkString = chunkString.split("\n");
        chunkString = chunkString.filter(item => Boolean(item?.trim?.()))
        chunkString = chunkString[chunkString.length - 1]
    
        chunkString = chunkString.substring("data: ".length - 1);

      onMessage(safeJSONParse(chunkString, chunkString));
      readChunk(reader, onFinish, onMessage);
    })
    .catch((error) => {
      console.error(error);
    });
};
