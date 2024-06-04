from graph.graph import Graph

def summarize_explanation(input, graph: Graph):
 
    inference = graph.get_memory("summarization_inference")

    result = inference.completion((
        "Your resposibility is to extract a list of key points in the following text "
        "that could be enhanced to generate a better response later"
        "Keep the list relevant and don't answer or provide details on your own\n"
        "RETURN A LIST IMMEDIATLY WITHOUT EXTRACT TEXT"
        "\n\n\n"
        "{explanation}"
    ).format_map(input))
 
    return {}
