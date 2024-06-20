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
            options = [SupportedToken.FULL.value, SupportedToken.PARTIAL.value, SupportedToken.NO_SUPPORT.value]
            response = inference.completion([
                {
                    "role": "system",
                    "content": (
                        "You are an artificial intelligence assistant in the medical field. "
                        "You will receive answer and an evidence for that answer,  from the user.\n"
                        "Your task is to evaluate if the answer is fully supported by the information provided in the evidence, and provide explanations on your judgement.\n"
                        "Use the following entailment scale to generate a score:\n"
                        "[Fully supported] - All information in answer is supported by the evidence, or extractions from the evidence."
                        "This is only applicable when the answer and part of the evidence are almost identical.\n"

                        "[Partially supported] - The answer is supported by the evidence to some extent, "
                        "but there is major information in the answer that is not discussed in the evidence."
                        " For example, if an instruction asks about two concepts and "
                        "the evidence only discusses either of them, it should be considered a [Partially supported].\n"

                        "[No support] - The answer completely ignores evidence, is unrelated to the evidence, "
                        "or contradicts the evidence. This can also happen if the evidence is irrelevant to the instruction.\n\n"
                        "Make sure to not use any external information/knowledge to judge whether the answer is true or not.\n"
                        "Only check whether the answer is supported by the evidence, and not whether the answer follows the instructions or not.\n"

                        "Respond only using one of the following options based on the criteria above: {options}"
                    ).format(options=["[{}]".format(option) for option in options])
                },

               
                {
                    "role": "user",
                    "content": (  
                        "answer: {output}"
                        "\n\n"
                        "evidence: {evidence}"
                        "\n\n"
                    ).format(input=input["query"], evidence="\n".join(input["documents"]), output=input["result"])
                }
            ])
 

            token = match_token(response, options)
            token = token.strip("[]")
 

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


def match_token(token: str, options: list[str]):
    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, token)

    if len(matches) > 0:
        for match in matches:
            if match in options:
                return "[{}]".format(match)

    for item in options:
        if item in token:
            return "[{}]".format(item)
