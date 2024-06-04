import { typewriter } from "@/lib/typewriter";
import { useEffect, useMemo, useState } from "react";

export function TypeWriter({ content, enabled }) {
  const [typingContent, setTypingContent] = useState("");

  const writer = useMemo(() => typewriter(), []);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    writer.type(content, setTypingContent);
  }, [content, enabled, writer]);

  return (
    <>
      {enabled ? (
        <>
          {typingContent}
          <span className="bg-red-500 w-0.5 inline-block h-[1rem] -mb-0.5 animate-blink" />
        </>
      ) : (
        content
      )}
    </>
  );
}
