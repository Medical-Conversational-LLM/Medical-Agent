from graph.graph import Graph

def stream_results(input, graph: Graph):
    graph.streamer.put({
        "type":"TOKEN",
        "message": input["result"]
    })
