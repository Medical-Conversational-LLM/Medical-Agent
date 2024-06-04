if __name__ == "__main__":
    import sys
    sys.path.append(".")

from run import create_graph
from threading import Thread
from utils import ThreadStreamer


def run_graph_concurrently(
    query: str,
    temperature=0.7,
    top_k=0.1,
    top_p=20,
    max_length=128
):
    graph = create_graph()
    graph.streamer = ThreadStreamer()

    thread = Thread(target=graph.start, args=(
        "check_retrieval", {"query": query}))

    thread.start()

    return graph.streamer.get()


if __name__ == "__main__":
    run_graph_concurrently(query="What is cancer")
