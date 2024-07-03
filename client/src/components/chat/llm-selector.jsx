import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Check } from "lucide-react"
import { Button } from "../ui/button"
import { useMemo } from "react"
const options = [
    {
        value: "rag",
        label: "Rag",
        description: "Integrated Retrieval-Augmented Generation"
    },
    {
        value: "self-reflective",
        label: "Self Reflective",
        description: "Reflective Enhanced Retrieval-Augmented Generation"
    },
    {
        value: "medline",
        label: "Medline",
        description: "Generation Integrated with Medline Dataset"
    }
]
export function LLMSelector({ className, selected = "self-reflective", onSelect }) {
    const selectedOption = useMemo(() => {
        return options.find(option => option.value === selected)
    }, [selected])
    return (
        <DropdownMenu className={className}>
            <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm">
                    {selectedOption?.label || selected}
                </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-72" align="end">
                {
                    options.map(option => (
                        <DropdownMenuItem className="relative p-3" onSelect={() => onSelect(option.value)}>
                            <div className="relative">
                                <div>{option.label}</div>
                                <div className="text-xs text-muted-foreground">{option.description}</div>

                            </div>
                            {
                                option.value === selected && (
                                    <div className="bg-primary text-primary-foreground  flex items-center justify-center rounded-full w-5 h-5 absolute top-2 right-2">
                                        <Check size={12} />
                                    </div>
                                )
                            }
                        </DropdownMenuItem>
                    ))
                }
            </DropdownMenuContent>
        </DropdownMenu>
    )
}