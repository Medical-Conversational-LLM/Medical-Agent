import { Button } from "@/components/ui/button";
import { useTheme } from "@/context/theme-provider";
import { Moon, Sun } from "lucide-react";
import { useEffect } from "react";

export function ThemeSwitch() {
  const { theme, setTheme } = useTheme();

  useEffect(() => {
    if(theme === "dark"){
      document.body.classList.add("dark")
    }else{
      document.body.classList.remove("dark")
    }
  }, [theme])
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
