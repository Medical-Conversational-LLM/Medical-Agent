from graph.graph import Graph
from prompt.supported import SupportedToken
MAX_ATTEMPTS = 0


def check_attempts(input, graph: Graph):
    input["continue"] = True
    attempts = graph.get_memory("attempts", 0)

    if attempts >= MAX_ATTEMPTS:

        if (graph.get_memory("utility") > 3
           and (
               graph.get_memory("supported") == SupportedToken.PARTIAL
               or graph.get_memory("supported") == SupportedToken.FULL
        )):

            return {
                **input,
                "result": graph.get_memory("result"),
            }

        return {
            **input,
            "result": input["result"] if "result" in input else "Sorry I'm not able to answer this question",
            "continue": False
        }
 
    graph.set_memory("attempts", attempts+1)

    return input
