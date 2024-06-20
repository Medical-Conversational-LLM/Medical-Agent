from graph.graph import Graph
from utils import extract_info_from_document, format_document_info_markdown


def stream_results(input, graph: Graph):
    
    print(input["documents"])
    graph.streamer.put({
        "type": "TOKEN",
        "message": format_document_info_markdown(input["result"], input["documents"])
    })
