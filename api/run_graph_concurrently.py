if __name__ == "__main__":
    import sys
    sys.path.append(".")

from lm.create_graph import create_graph
from threading import Thread
from utils import ThreadStreamer


def run_graph_concurrently(
    query: str,
    chat_history,
    temperature=0.7,
    top_k=0.1,
    top_p=20,
    max_length=128
):
    graph = create_graph()
    graph.streamer = ThreadStreamer()

    input_data = {
        "query": query,
        "chat_history": chat_history
    }

    thread = Thread(target=graph.start, args=("check_retrieval", input_data))

    thread.start()

    return graph.streamer.get()


if __name__ == "__main__":
    run_graph_concurrently(query="What is cancer")
