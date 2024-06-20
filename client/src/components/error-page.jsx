import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"
import axios from "axios"
import { useEffect, useState } from "react"
import { OctagonAlert } from "lucide-react"
export function ErrorPage() {

    const [error, setError] = useState()

    useEffect(() => {
        axios.interceptors.response.use(
            response => response,
            error => {
                setError({
                    url: error.response.request.responseURL,
                    status: error.response.status,
                    statusText: error.response.statusText
                })
                console.log("error.response.status", error.response)
            });
    }, [])
    return (
        <Dialog open={!!error} onOpenChange={() => setError(undefined)}>
            <DialogContent className="flex gap-6">
                <div className="w-12 h-12 bg-red-50 text-red-500 flex items-center justify-center rounded-full shrink-0">
                    <OctagonAlert size={30} />
                </div>
                <DialogHeader className="flex-1">
                    <DialogTitle className="mb-2">Something went wrong</DialogTitle>
                    <DialogDescription>
                        <div className="flex flex-col gap-1 mb-4">
                        <div>
                            Request URL: <Badge>
                                {error?.url}
                            </Badge>
                        </div>
                        <div>
                            Status: <Badge>
                                {error?.statusText} ({error?.status})
                            </Badge>
                        </div>
                        </div>
                        <div className="font-bold">
                            Don’t worry, it’s not your fault! Please try the following:
                        </div>
                        <div>
                            •	Refresh the page and try again.
                            <br />
                            •	Check your internet connection.
                            <br />
                            •	If the problem persists, please contact support.
                        </div>
                        <br />
                        <div>
                            We apologize for the inconvenience and appreciate your patience.
                        </div>
                    </DialogDescription>
                </DialogHeader>
            </DialogContent>
        </Dialog>
    )
}