from graph.graph import Graph

def critique_relevant(input, graph: Graph):

    graph.streamer.put({
        "type": "CRITIQUE_RELEVANT",
        "message": "Critique if the response is relevant"
    })
 
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
        prompt
    )

    if(result == "[Irrelevant]"):
        return []
    
    return input["documents"]
