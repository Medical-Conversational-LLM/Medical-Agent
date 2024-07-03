from utils import get_llama_formatted_prompt

from typing import Any
from graph.graph import Graph
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompt.supported import SupportedToken
import re
from prompt.supported import Supported, SupportedToken

CRITIQUE_WEIGHTS = {
    'ISREL': 1.0,
    'ISSUP': 1.0,
    'ISUSE': 1.0
}
K = 2  
B = 2


def generate_candidates_for_chunk(input, inference, chunk=None, chat_history=[]):

    chat_history_string = "\n".join(["{role}: {content}".format(
        role=item["role"].title(), content=item["content"]) for item in chat_history])

    history_prompt = {
        "role": "system",
        "content": (
            "{pre_history}"
            "{history}"
        ).format(
            pre_history=(
                "Please respect the following conversation history tagged by User and Assistant, "
                "continuing the dialogue in a logical and consistent manner based on the previous exchanges"
            ) if len(chat_history) > 0 else "",
            history=chat_history_string
        )
    }
    if chunk is not None:
        prompt = [
            {
                "role": "system",
                "content": (
                    "This is a chat between a user and an artificial intelligence assistant in the medical field.\n "
                    "The assistant gives helpful, detailed answers to the user's questions based on the provided context.\n"
                )
            },
            history_prompt,
            {
                "role": "user",
                "content": (
                    "##Question:\n{question}\n"
                    "##Context: {context}\n"
                    "##Answer:"
                ).format(
                    question=input["query"],
                    context=chunk,
                )
            }
        ]

    else:

        prompt = [
            {
                "role": "system",
                "content": (
                    "This is a chat between a user and an artificial intelligence assistant in the medical field.\n "
                    "The assistant gives helpful, detailed answers to the user's questions based on previous knowledge .\n"
                )
            },
            history_prompt,
            {
                "role": "user",
                "content": (
                    "##Question:\n{question}\n"
                    "##Answer:"
                ).format(
                    question=input["query"],
                )
            }
        ]
    print('=== prompt ===' , prompt)

    result = inference.completion(prompt)

    return result


def generate(input, graph: Graph):

    inference = graph.get_memory("generator_inference")

    graph.streamer.put({
        "type": "GENERATE",
        "message": "Generating response"
    })

    chunks = input["documents"].copy()
    if len(chunks) > 2 :
       chunks = chunks[0:2]

    chat_history = []
    for item in input['chat_history']:
        role = "Assistant"

        if "user_id" in item and item["user_id"] is not None:
            role = "User"

        chat_history.append({
            "role": role,
            "content": item["content"]
        })

    all_candidates = []

    with ThreadPoolExecutor(max_workers=6) as executor:
        if isinstance(chunks, list) is False or len(chunks) == 0:
            chunks = [None]

        futures = [executor.submit(
            generate_candidates_for_chunk, input, inference, chunk, chat_history) for chunk in chunks]

        for future in as_completed(futures):
            all_candidates.append(future.result())

    scored_candidates = []
    i = 0
    for candidate in all_candidates:

        graph.streamer.put({
            "type": "CRITIQUE",
            "message": "Critique candidate {} relevance".format(i)
        })

        i += 1

        relevance_score = critique_relevant(input, graph)
        graph.streamer.put({
            "type": "CRITIQUE",
            "message": "Critique candidate {} groundness".format(i)
        })
        support_score = critique_supported(input, candidate, graph)
        graph.streamer.put({
            "type": "CRITIQUE",
            "message": "Critique candidate {} usefulness".format(i)
        })
        utility_score = critique_utility(input, candidate, graph)
        print('scores',relevance_score,support_score,utility_score)

        S_Critique = (
            CRITIQUE_WEIGHTS['ISREL'] * int(relevance_score) +
            CRITIQUE_WEIGHTS['ISSUP'] * int(support_score) +
            CRITIQUE_WEIGHTS['ISUSE'] *
            utility_score if type(utility_score) is int else 0
        )
 
        lm_probability = 0.6
 
        segment_score = lm_probability + S_Critique
        scored_candidates.append(
            {'candidate': candidate, 'score': segment_score})
        
    top_segments = beam_search(scored_candidates)
 
    final_result = top_segments[0]['candidate'] if top_segments else "No valid continuation found."
    
    return final_result


def critique_relevant(input, graph: Graph):

    context = "\n".join(input["documents"])

    prompt = [
        {
            "role": "system",
            "content": (
                "This is a chat between a user and an artificial intelligence assistant."
                "The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. "
                "The assistant should also indicate when the answer cannot be found in the context.\n"
                "When given input and evidence, evaluate whether the evidence is relevant to the input "
                "and provides valuable information for generating meaningful responses.\n",
                "Use a rating of [Relevant] to indicate relevance and usefulness, and [Irrelevant] to indicate irrelevance."
            )
        },
        {
            "role": "user",
            "content": (
                "Input: {input}"
                "\n\n"
                "Evidance :{context}"
                "\n\n"
            ).format(input=input["query"], context=context)
        }
    ]

    inference = graph.get_memory("relevance_inference")

    result = inference.completion(
        prompt,
        max_new_tokens=15
    )

    if (result == "[Irrelevant]"):
        return 0

    return 1


def critique_utility(input, output, graph: Graph):

    def utility_to_score(utility): 
        utility_score_mapping = {
            5: 1.0,
            4: 0.75,
            3: 0.5,
            2: 0.25,
            1: 0.0
        }
 
        return utility_score_mapping.get(utility, "Invalid utility rating")

    inference = graph.get_memory("critique_utility_inference")
    options = ['Utility:5', 'Utility:4', 'Utility:3', 'Utility:2', 'Utility:1']
    prompt = [
        {
            "role": "system",
            "content": (
                "This is a chat between a user and an artificial intelligence assistant in the medical field. "
                "The assistant gives Utility score response based on the options criteria.\n"
            ),
        },
        {
            "role": "user",
            "content": (
                "Given an input and an output, rate whether the output appears to be a helpful and informative answer to the input query, "
                "from 1 (lowest) - 5 (highest). We call this score perceived utility.\n"
                "[Utility:5]: The response provides a complete,highly detailed, and informative response to the query, fully satisfying the information needs.\n"
                "[Utility:4]: The response mostly fulfills the need in the query,while there can be some minor improvements such as discussing more detailed information,having better structure of the response, or improving coherence. \n"
                "[Utility:3]: The response is acceptable, but some major additions or improvements are needed to satisfy users' needs.\n"
                "[Utility:2]: The response still addresses the main request, but it is not complete or not relevant to the query.\n"
                "[Utility:1]: The response is barely on-topic or completely irrelevant.\n"
                "Respond only with score perceived utility using one of the following options based on the criteria above: {options}. \n"
            ).format(options=["[{}]".format(option) for option in options])
        },
        {
            "role": "user",
            "content": (
                "Input: {input}"
                "\n\n"
                "Output: {output}"
                "\n\n"
            ).format(input=input["query"], evidence="\n\n".join(input["documents"]), output=output)
        }
    ]

    def extract_utility_number(text):
        match = re.search(r'\[Utility:(\d+)\]', text)
        if match:
            return int(match.group(1))
        else:
            return None

    response = inference.completion(prompt, max_new_tokens=15)
    utility_number = extract_utility_number(response)
    score = utility_to_score(utility_number)
    return score


def critique_supported(input, output, graph: Graph):

    for i in range(0, 2):
        try:
            inference = graph.get_memory("critique_supported_inference")
            options = [SupportedToken.FULL.value,
                       SupportedToken.PARTIAL.value, SupportedToken.NO_SUPPORT.value]
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
                    ).format(input=input["query"], evidence="\n".join(input["documents"]), output=output)
                }
            ], max_new_tokens=25)

            token = '[Fully supported]'
            token = match_token(response, options)
            if token is not None:
                token = token.strip("[]")

            if token == 'Fully supported':
                return 1
            elif token == 'Partially supported':
                return 0.5
            else:
                return 0
        except Exception as e:
            print("error")
            print(e)

            pass
        return 0


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


def beam_search(segments):
    segments.sort(key=lambda x: x['score'], reverse=True)
    if len(segments) >= B:
        return segments[:B]
    else:
        return segments