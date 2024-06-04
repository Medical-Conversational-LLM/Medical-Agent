import { useCallback, useEffect, useRef } from "react";

export const useKeyPress = (ref) => {
  const keys = useRef({});
  useEffect(() => {
    if (!ref.current) {
      return;
    }
    const element = ref.current;

    const onPress = (e) => {
      keys.current[e.key] = true;
    };

    const onUp = (e) => {
      delete keys.current[e.key]
    };

    element.addEventListener("keydown", onPress);
    element.addEventListener("keyup", onUp);
    return () => {
      element.removeEventListener("keydown", onPress);
      element.removeEventListener("keyup", onUp);
    };
  }, []);

  return useCallback((key) => {
    console.log("keys.current", keys.current)
    return keys.current[key];
  }, []);
};
