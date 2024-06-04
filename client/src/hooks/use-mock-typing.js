import { useEffect } from "react";

const content = `Hello Dear reader
  Hello Dear reader
  Hello Dear reasdsdsdder
  Hello Dear rea2323 der
  Hello Dear rea23der`
  .split("\n")
  .flatMap((word) => [word, " "]);

content.pop();

export function useMockTyping({
  message,
  updateMessages,
  enabled = false,
  listRef,
  onFinish,
  onCancel,
}) {
  useEffect(() => {
    if (!enabled) {
      return;
    }

    const promises = content.map((word, index) => {
      return wait(() => {
        updateMessages([
          {
            ...message,
            content: message.content + word,
          },
        ]);

        requestAnimationFrame(() => {
          listRef.current.scrollToEnd();
        });
      }, 1000 * index);
    });

    Promise.all(promises)
      .then(onFinish)
      .catch(() => console.log("cancelled mocked data"));

    return () => {
      for (let promise of promises) {
        promise.cancel();
      }
      onCancel();
    };
  }, [enabled, updateMessages, listRef, onFinish, message, onCancel]);
}

const wait = (callback, time) => {
  const controller = new AbortController();

  const promise = new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      callback();
      resolve();
    }, time);

    const abortListener = ({ target }) => {
      controller.signal.removeEventListener("abort", abortListener);
      clearTimeout(timer);
      reject(target.reason);
    };
    controller.signal.addEventListener("abort", abortListener);
  });

  promise.cancel = controller.abort.bind(controller);

  return promise;
};
