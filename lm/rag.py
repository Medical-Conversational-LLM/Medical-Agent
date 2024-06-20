from inference.inference import Inference
from graph.graph import Graph

                    # "prediction should followed by a source of your generated response / knowledge , including the source name, publication date, or URL .\n"
def rag(input, graph: Graph):

    inference = graph.get_memory("generator_inference")
    chunks = input["documents"]
    chunks = chunks[0:5]

    chat_history = []
    for item in input['chat_history']:
        role = "Assistant"

        if "user_id" in item and item["user_id"] is not None:
            role = "User"

        chat_history.append({
            "role": role,
            "content": item["content"]
        })

    result = generate_candidates_for_chunk(
        input, inference, "\n\n".join(chunks), chat_history)

    return result


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
                    "The assistant gives helpful, detailed answers to the user's questions based on the provided context."
                    "At the end of the generate response, provide the evidence source URLs from each context you used to in the response , Please follow the example below "
                    
                    """
                    Example:\n
                    ###User: What are the treatment options for diabetes?\n
                    ###Context (with URLs):There are many diabetes treatments available to help you manage your condition. Everyone is different, so treatment will vary depending on your own individual needs. Not all diabetes treatments are suitable for everyone, so don’t be disheartened if you find yourself needing to change or stop certain medications. Your GP or healthcare team can help you find a medication that’s best for you.\n
                    ###Response:Diabetes mellitus is a chronic condition that affects how your body uses blood sugar (glucose). Treatment options include lifestyle changes, medications, and insulin therapy. Learn more at:
                    [Source: https://www.mayoclinic.org/]
                    """
                )
            },
            history_prompt,
            {
                "role": "user",
                "content": (
                    "##Question:\n{question}\n"
                    "##Context: {context}\n"
                    "##Response:"
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
                    "##Response:"
                ).format(
                    question=input["query"],
                )
            }
        ]

    prompt = prompt

    result = inference.completion(prompt)

    return result
