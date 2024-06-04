from prompt.prompt import Prompt
from typing import Any
from utils import create_training_object

UTILITY_EXPLAINATION_SUMMARY_TITLE = (
    "the rating of whether the response appears to be a "
    "helpful and informative answer to the query, "
    "from 1 (lowest) - 5 (highest)"
)

OUTPUT_INSTRUCTIONS = (
    "Given an Input and an Output, rate whether the response appears to be a helpful and informative answer to the query, from 1 (lowest) - 5 (highest). We call this score perceived utility.\n[Utility:5]: The response provides a complete, highly detailed, and informative response to the query, fully satisfying the information needs.Do not provide explanations on your judgement\n"
    "[Utility:4]: The response mostly fulfills the need in the query, while there can be some minor improvements such as discussing more detailed information, having better structure of the response, or improving coherence. \n"
    "[Utility:3]: The response is acceptable, b ut some major additions or improvements are needed to satisfy users' needs.\n"
    "[Utility:2]: The response still addresses the main request, but it is not complete or not relevant to the query.\n"
    "[Utility:1]: The response is barely on-topic or completely irrelevant.\n"
)

PROMPT_DICT = {
    "context": (
        "instruction :Given an Input and an Output, rate whether the response appears to be a helpful and informative answer to the query, from 1 (lowest) - 5 (highest). We call this score perceived utility.\n[Utility:5]: The response provides a complete, highly detailed, and informative response to the query, fully satisfying the information needs.\n"
        "Please provide short explanation on your judgement max (10 words)\n"
        "[Utility:4]: The response mostly fulfills the need in the query, while there can be some minor improvements such as discussing more detailed information, having better structure of the response, or improving coherence. \n"
        "[Utility:3]: The response is acceptable, but some major additions or improvements are needed to satisfy users' needs.\n"
        "[Utility:2]: The response still addresses the main request, but it is not complete or not relevant to the query.\n"
        "[Utility:1]: The response is barely on-topic or completely irrelevant.\n"
        "Example 1:\n"
        "Input: What are the common symptoms of the common cold?\n"
        "Output: The common symptoms of the common cold include runny or stuffy nose, sore throat, cough, congestion, slight body aches, and mild fatigue.\n"
        "Response : [Utility: 5] \n\n"
        "Example 2: \n"
        "Input: Describe the treatment options for mild to moderate asthma.\n"
        "Output: Treatment options for mild to moderate asthma include inhaled corticosteroids and bronchodilators.\n"
        "Response : [Utility: 3] \n\n"
        "USER INPUT: \n\n"
        "Input: {input}\n"
        "Output: {Output}\n"
        "Use the following template as response\n"
        "Response:\n [Utility:*]"
        "\n\nExplanation:\n\n"
    ),

}


def create_result(output):
    return {"output": output}


class Utility(Prompt):

    def format(self, data: dict[str, str]) -> str:
        return PROMPT_DICT["context"].format_map(data)

    def parse(self, result: str, row: Any) -> str:

        # try:
        #     output = int(result.split(":"))
        # except:
        #     pass

        # if output is None:
        #     return None

        return {"token": "[{}]".format(result), 'instruction': OUTPUT_INSTRUCTIONS, "task": 'utility',
                "input": row["input"] + " \nOutput:" + row["Output"]}

        return create_training_object(
            OUTPUT_INSTRUCTIONS,
            ("Task instruction: {instruction}\n" "Output: {output}").format_map(
                row),
            "[{}]".format(score),
            "utility",
        )
