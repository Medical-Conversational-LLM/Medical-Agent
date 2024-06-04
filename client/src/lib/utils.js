import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

export function wait(timeMS) {
  return new Promise((resolve) => setTimeout(resolve, timeMS));
}

export function safeJSONParse(json, defaultValue) {
  try {
    return JSON.parse(json);
  } catch (error) {
    return defaultValue;
  }
}
