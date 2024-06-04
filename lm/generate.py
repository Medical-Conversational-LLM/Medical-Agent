from inference.groq import Groq
from graph.graph import Graph


def generate(input, graph: Graph):
    inference = graph.get_memory("generator_inference")

    prompt = [
        {
            "role": "system",
            "content": (
                "This is a chat between a user and an artificial intelligence assistant in the medical field. "
                "The assistant gives helpful, detailed answers to the user's questions based on the provided context.\n"
            )
        },
        {
            "role": "user",
            "content": (
                "##Question:\n{question}\n"
                "##Context: {context}\n"
                "##Response:"
            ).format(
                question=input["query"],
                context=input["documents"]
            )
        }
    ]
    graph.streamer.put({
        "type": "GENERATE",
        "message": "Generating initial response"
    })
 

    result = inference.completion(
        prompt
    )

    print(
        result
    )

    return result
