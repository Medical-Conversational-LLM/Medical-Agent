if __name__ == "__main__":
    import sys
    sys.path.append(".")

from lm.create_graph import create_graph, create_rag_graph
from threading import Thread
from utils import ThreadStreamer


def run_graph_concurrently(
    query: str,
    chat_history,
    temperature=0.7,
    top_k=0.1,
    top_p=20,
    max_length=128,
    model: str = "self-reflective"
):

    input_data = {
        "query": query,
        "chat_history": chat_history
    }

    entry = "check_retrieval"

    if model == "rag":
        entry = "vector_db"
        graph = create_rag_graph()
    else:
        graph = create_graph()

    print(f"RUNNING model {model} ->entry-> [{entry}] ")
    
    graph.streamer = ThreadStreamer()

    thread = Thread(target=graph.start, args=(entry, input_data))

    thread.start()

    return graph.streamer.get()


if __name__ == "__main__":
    run_graph_concurrently(query="What is cancer")
