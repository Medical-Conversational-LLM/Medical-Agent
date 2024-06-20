from prompt.supported import SupportedToken
from graph.graph import Graph
from utils import match_reflective_token_with_explaination
import re


def critique_utility(input, graph: Graph):

    graph.streamer.put({
        "type": "CRITIQUE_UTILITY",
        "message": "Critique if the response is Useful"
    })

    inference = graph.get_memory("critique_utility_inference")

    prompt = [
        {
            "role": "system",
            "content": (
                "This is a chat between a user and an artificial intelligence assistant in the medical field. "
                "The assistant gives helpful, detailed answers to the user's questions.\n"
            ),
        },
        {
            "role": "user",
            "content": (
                "Given an input and an output, rate whether the response appears to be a helpful and informative answer to the query, "
                "from 1 (lowest) - 5 (highest). We call this score perceived utility.\n[Utility:5]: The response provides a complete,"
                "highly detailed, and informative response to the query, fully satisfying the information needs.\n"
                "[Utility:4]: The response mostly fulfills the need in the query, "
                "while there can be some minor improvements such as discussing more detailed information, "
                "having better structure of the response, or improving coherence. \n"
                "[Utility:3]: The response is acceptable, but some major additions or improvements are needed to satisfy users' needs.\n"
                "[Utility:2]: The response still addresses the main request, but it is not complete or not relevant to the query.\n"
                "[Utility:1]: The response is barely on-topic or completely irrelevant.\n"
            )
        },
        {
            "role": "user",
            "content": (
                "Input: {input}"
                "\n\n"
                "Output: {output}"
                "\n\n"
                "Evidence :{evidence}"
                "\n\n"
            ).format(input=input["query"], evidence="\n\n".join(input["documents"]), output=input["result"])
        }
    ]
 
    response = inference.completion(prompt)

    match = re.search(r'\d+', response)

    score = 4
    if match:
        score = int(match.group())

    graph.set_memory("utility", score)
    graph.set_memory("supported", input["token"])
    graph.set_memory("result", input["result"])
 
    return {
        "score": score,
        "explanation": ""
    }
