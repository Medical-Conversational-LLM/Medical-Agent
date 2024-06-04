import { useEffect, useState } from "react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Slider } from "../ui/slider";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { createPortal } from "react-dom";
import { Settings } from "lucide-react";
import { cn } from "@/lib/utils";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

export function Customize({ settings, onUpdate }) {
  const [portalTarget, setPortalTarget] = useState();
  useEffect(() => {
    setPortalTarget(document.getElementById("layout-actions"));
  }, []);

  const content = (
    <div className="grid gap-6">
      <SliderInput
        label="Temperature"
        value={settings.temperature}
        onChange={(value) => onUpdate("temperature", value)}
        description="Controls the randomness of the generated output. A higher temperature value, such as 1.0, leads to more randomness and diversity in the generated text"
      />
      <SliderInput
        label="Top P"
        value={settings.top_p}
        onChange={(value) => onUpdate("top_p", value)}
        max={1}
        step={0.01}
        description="Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered."
      />
      <SliderInput
        label="Top K"
        value={settings.top_k}
        onChange={(value) => onUpdate("top_k", value)}
        description="Top-k sampling picks from the k most likely words for generating text, ensuring quality and variety"
      />
      <SliderInput
        label="Max Length"
        max={4096}
        step={1}
        value={settings.max_length}
        onChange={(value) => onUpdate("max_length", value)}
        description="the maximum number of tokens, or words, allowed in the output of the generated text."
      />
    </div>
  );
  return (
    <>
      <div
        className={cn(
          "hidden lg:block w-64 h-fit  gap-6 max-h-[calc(100vh-10rem)] p-4 self-center border mr-10 shadow-sm rounded-xl bg-background "
        )}
      >
        {content}
      </div>
      {!!portalTarget &&
        createPortal(
          <Dialog>
            <DialogTrigger asChild>
              <Button
                variant="ghost"
                className="rounded-full lg:hidden"
                size="icon"
              >
                <Settings />
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogTitle>Chat settings</DialogTitle>

              <DialogHeader>{content}</DialogHeader>
            </DialogContent>
          </Dialog>,
          portalTarget
        )}
    </>
  );
}

const SliderInput = ({
  label,
  value,
  onChange,
  step = 0.2,
  max = 2,
  description = "",
}) => {
  return (
    <HoverCard openDelay={200} closeDelay={0}>
      <HoverCardTrigger>
        <div className="grid gap-2">
          <Label className="flex justify-between items-center">
            {label}
            <Input
              size="sm"
              className="w-14"
              value={value}
              onChange={(e) => {
                if (isNaN(Number(e.target.value))) {
                  return;
                }
                onChange(Number(e.target.value));
              }}
            />
          </Label>
          <Slider
            value={[value]}
            max={max}
            step={step}
            onValueChange={(value) => {
              onChange(value[0]);
            }}
          />
          <div className="text-start text-xs text-muted-foreground lg:hidden">
          {description}
          </div>
        </div>
      </HoverCardTrigger>
      <HoverCardContent side="left" sideOffset={35} className="max-md:hidden">
        {description}
      </HoverCardContent>
    </HoverCard>
  );
};
