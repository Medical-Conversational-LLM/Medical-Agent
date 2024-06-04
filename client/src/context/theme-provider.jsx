import { cn } from "@/lib/utils";
import { createContext, useContext, useState } from "react";

const Context = createContext({
  theme: "light",
});

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState("light");

  const updateTheme = (theme) => {
    setTheme(theme);
  };

  return (
    <Context.Provider value={{ theme, setTheme: updateTheme }}>
      <div className={cn(theme, "text-foreground text-sm")}>
        {children}
        
      </div>
    </Context.Provider>
  );
}

export const useTheme = () => useContext(Context);
