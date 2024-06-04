import { Button } from "@/components/ui/button";
import { useTheme } from "@/context/theme-provider";
import { Moon, Sun } from "lucide-react";

export function ThemeSwitch() {
  const { theme, setTheme } = useTheme();

  return (
    <Button
      className="rounded-full"
      size="icon"
      variant="ghost"
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
    >
      {theme === "dark" && <Moon />}
      {theme !== "dark" && <Sun />}
    </Button>
  );
}
