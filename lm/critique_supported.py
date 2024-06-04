from prompt.supported import Supported, SupportedToken
from graph.graph import Graph
import re

def critique_supported(input, graph: Graph):
    
    graph.streamer.put({
        "type": "CRITIQUE_SUPPORTED",
        "message": "Critique if the response is supported by the documents"
    })
    
    for i in range(0, 2):
        try:
            inference = graph.get_memory("critique_supported_inference")

            response = inference.completion([
                {
                    "role":"system",
                    "content": (
                        "You will receive an input, evidence, and output.\n"
                        "Your task is to evaluate if the output is fully supported by the information provided in the evidence, and provide explanations on your judgement.\n"
                        "Use the following entailment scale to generate a score:\n"
                        "[Fully supported] - All information in output is supported by the evidence, or extractions from the evidence."
                        "This is only applicable when the output and part of the evidence are almost identical.\n"

                        "[Partially supported] - The output is supported by the evidence to some extent, "
                        "but there is major information in the output that is not discussed in the evidence."
                        " For example, if an instruction asks about two concepts and "
                        "the evidence only discusses either of them, it should be considered a [Partially supported].\n"

                        "[No support] - The output completely ignores evidence, is unrelated to the evidence, "
                        "or contradicts the evidence. This can also happen if the evidence is irrelevant to the instruction.\n\n"
                        "Make sure to not use any external information/knowledge to judge whether the output is true or not.\n"
                        "Only check whether the output is supported by the evidence, and not whether the output follows the instructions or not.\n"
                        
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
            ])
  
            token = match_token(response, [SupportedToken.FULL.value, SupportedToken.PARTIAL.value, SupportedToken.NO_SUPPORT.value])
            token = token.strip("[]")

            print(token)
 
            return {
                "token": token,  "explanation": ""
            }
        except Exception as e:
            print("error")
            print(e)

            pass

        return {
            "token": SupportedToken.NO_SUPPORT,  "explanation": ""
        }


def match_token(token:str, options: list[str]):
    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, token)

    if len(matches) > 0:
        for match in matches:
            if match in options:
                return "[{}]".format(match)
        
    for item in options:
        if item in token:
            return "[{}]".format(item)
