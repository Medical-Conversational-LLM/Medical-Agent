import { typewriter } from "@/lib/typewriter";
import { cn } from "@/lib/utils";
import { useEffect, useMemo, useState } from "react";
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card"

export function TypeWriter({ content, enabled }) {
  const [typingContent, setTypingContent] = useState("");

  const writer = useMemo(() => typewriter(), []);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    writer.type(content, setTypingContent);
  }, [content, enabled, writer]);


  const output = <Markdown
    components={{
      code(props) {
        const { children, className, node, ...rest } = props

        console.log("children", children)
        const match = /loe/.exec(className || '')
        return match ? (
          <div> 
            <LevelOfEvidence selected={children} />
          </div>
        ) : (
          <code {...rest} className={className}>
            {children}
          </code>
        )
      }
    }}

    remarkPlugins={[remarkGfm]} children={enabled ? typingContent : content} />


  return (
    <div className="prose" >
      {
        enabled ? (
          <>
            {output}
            < span className="bg-red-500 w-0.5 inline-block h-[1rem] -mb-0.5 animate-blink" />
          </>
        ) : (
          output
        )
      }
    </div >
  );
}


const LevelOfEvidence = ({ selected }) => {

  selected = selected?.replace?.("\n", "")?.trim?.()


  if(!selected) return null

  const levels = [
    {
      label: "1a",
      value: 0,
      color: "#2D7E43",
      description:
        "Background information on the topic, sourced from systematic reviews of randomized controlled trials.",

    },
    {
      label: "1b",
      value: 1,
      description: "Data derived from individual randomized controlled trials.",
      "color": "#97BA38"
    },
    {
      label: "2a",
      value: 2,
      description: "Insights from systematic reviews of cohort studies.",
      "color": "#F0CA0D"
    },
    {
      label: "2b",
      value: 3,
      description:
        "Details from individual cohort studies or low-quality randomized controlled trials.",
      "color": "#D57B1A"
    },
    {
      label: "3a",
      value: 4,
      description: "Information from systematic reviews of case-control studies.",
      "color": "#C53419"
    },
    {
      label: "3b",
      value: 5,
      description: "Data from individual case-control studies.",
      "color": "#821D0A"
    },
 
    {
      label: "4",
      value: 6,
      color:"#62190C",
      description:
        "Observations from case series or poor quality cohort and case-control studies.",
    },



  ];

  return (
    <div className="flex ring ring-background/10 inline-flex rounded-md">
      {
        levels.map((level, i) => (

          <HoverCard openDelay={50}>
            <HoverCardTrigger
              style={{
                background: level.color
              }}
              className={cn(
                i === 0 && "rounded-l-md",
                i === levels.length - 1 && "rounded-r-md",
                "w-10 h-9 flex items-center justify-center font-medium text-lg font-sans no-underline text-white",
                String(level.value) === String(selected) && " rounded-md ring ring-white scale-110"
              )}
            >
              {level.label}

            </HoverCardTrigger>
            <HoverCardContent className="font-sans whitespace-break-spaces	">
              {level.description}
            </HoverCardContent>
          </HoverCard>


        ))
      }
    </div>
  )
}