from graph.graph import Graph
import re


def check_need_retrieval(input, graph: Graph):

    if "query" not in input or input["query"] is None:
        raise ValueError("query is required")

    inference = graph.get_memory("retrieval_inference")

    prompt = [
        {
            "role": "system",
            "content":re.sub(r'\n', '', (
                    "This is a chat between a user and an artificial intelligence assistant."
                    "Your task is to determine if the question necessitates acquiring supplementary information "
                    "from external sources for a more precise response."
                    "If additional information is needed return [Retrieval], if it is not return [No Retrieval]\n"
                    "# Examples:\n\n"
                    "Question: What are the common symptoms of type 2 diabetes\n"
                    "RetrievalOrNot: [No Retrieval]\n"
 
                    "Question: What are the latest FDA-approved treatments for advanced melanoma, and how do they compare in terms of efficacy and safety?\n"
                    "RetrievalOrNot: [Retrieval]\n"
 
                    "Ensure to return results similiar to the examples structure"
            ))
        },
        {
            "role": "user",
            "content": re.sub(r'\n', '',(
                "Question:{query}\n"
                "RetrievalOrNot:\n" 
            ).format(query=input["query"]))
        }
    ]

    model_response = inference.completion(prompt)
    model_response = model_response.strip().lower()
     
    return "no retrieval" not in model_response
