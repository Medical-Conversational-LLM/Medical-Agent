from graph.graph import Graph
def check_need_retrieval(input, graph:Graph):
    inference = graph.get_memory("retrieval_inference")
    prompt = [
        {
            "role": "system",
            "content": "This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
        },
        {
            "role": "user",
            "content": f"""Your task is to evaluate whether the question requires seeking additional information from external sources to produce a more comprehensive and accurate response ,return [Retrieval] or [No Retrieval]"""
        },
        {
            "role": "user", "content": "Describe the symptoms of COVID-19?"
        }
    ]

    result = inference.completion(prompt)
    print('Retrieval =' , result)

    if "query" not in input or input["query"] is None:
        raise ValueError("query is required")
    
    return result != "[No Retrieval]"