export const typewriter = ({ duration = 5 } = {}) => {
  let currentText = "";
  let fullText = "";
  let interval;
  let position = 0;

  return {
    type(text, callback) {
      fullText = text;
      clearInterval(interval);
      const speed = text.length == 0 ? 0 : duration / text.length;

      if (speed === 0) {
        callback(text);
        return;
      }

      interval = setInterval(() => {
        if (position >= text.length - 1) {
          return;
        }
        currentText += text[position];
        callback(currentText);

        position += 1;
      }, speed * 1000);
    },
    finish(callback) {
      if (typeof callback === "function") {
        callback(fullText);
      }
      clearInterval(interval);
    },
  };
};
