from graph.graph import Graph
from lm.rag import rag
from lm.check_need_retrieval import check_need_retrieval
from lm.retriever.elasticsearch import elasticsearch_node
from lm.retriever.web_search import web_search
from lm.retriever.vector_db import vector_db
from lm.critique_supported import critique_supported
from lm.stream_results import stream_results
from lm.generate import generate
from lm.critique_relevant import critique_relevant
from lm.critique_utility import critique_utility
from lm.summarize_explanation import summarize_explanation
from lm.check_attempts import check_attempts
from inference.groq import Groq
from inference.pretrained import Pretrained
from prompt.supported import SupportedToken
from utils import list_available_devices, should_continue
# inference = Groq(
#     api_key="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",
#     model="llama3-8b-8192",
# )
try:
    critique_inference = Pretrained("HlaH/Llama3-ChatQA-Critic-PubMedQA")
    retrieval_inference = Pretrained("HlaH/Llama3-ChatQA-Retriever-PubMedQA")
    generator_inference = Pretrained("HlaH/Llama3-ChatQA-Generator-PubMedQA")
except Exception as error:
    list_available_devices()
    raise error
MIN_RELEVANT_TO_WEB_SEARCH = 6
MIN_AVG_TO_WEB_SEARCH = 0.3


# summarization_inference = inference
def generate_with_history(input):
    # input should contain 'chat_history'
    return generate(input['query'], input['documents'])


def setup_graph_memory(graph: Graph):
    graph.set_memory("retrieval_inference", retrieval_inference)
    graph.set_memory("relevance_inference", critique_inference)
    graph.set_memory("critique_supported_inference", critique_inference)
    graph.set_memory("critique_utility_inference", critique_inference)
    graph.set_memory("generator_inference", generator_inference)


def create_graph():

    graph = Graph()

    setup_graph_memory(graph)

    graph.add_node("check_retrieval", check_need_retrieval)
    graph.add_node("vector_db", vector_db)
    graph.add_node("elasticsearch", elasticsearch_node)
    graph.add_node("web_search", web_search)
    graph.add_node("critique_relevant", critique_relevant)

    graph.add_node("stream_results", stream_results)
    graph.add_node("generate", generate)

    graph.add_edge(
        "check_retrieval",
        "vector_db",
        condition=True,
        out=lambda input: input
    )
    graph.add_edge("check_retrieval", "generate", condition=False)

    # graph.add_edge(
    #     "vector_db",
    #     "web_search",
    #     condition=lambda result: len(result['documents']) < MIN_RELEVANT_TO_WEB_SEARCH,
    #     out=lambda input, result: ({**input, **result}),
    # )

    graph.add_edge(
        "vector_db",
        "generate",
        out=lambda input, result: ({**input, "documents": result}),
        condition=lambda result: len(result) > 0,
        description="docs > 0"
    )

    graph.add_edge("generate", "stream_results",
                   out=lambda input, result: ({
                       **input,
                       "result": result
                   }))

    return graph


def create_rag_graph():

    graph = Graph()

    setup_graph_memory(graph)

    graph.add_node("vector_db", vector_db)
    graph.add_node("web_search", web_search)
    graph.add_node("critique_relevant", critique_relevant)
    graph.add_node("elasticsearch", elasticsearch_node)

    graph.add_node("stream_results", stream_results)
    graph.add_node("rag", rag)

    graph.add_edge(
        "check_retrieval",
        "vector_db",
        condition=True,
        out=lambda input: input
    )
    
    graph.add_edge("check_retrieval", "rag", condition=False)

    graph.add_edge(
        "vector_db",
        "elasticsearch",
        condition=lambda result: len(result['documents']) < MIN_RELEVANT_TO_WEB_SEARCH or
        result['avg'] > MIN_AVG_TO_WEB_SEARCH,
        out=lambda input, result: ({**input, **result}),
        description="Docs < {}".format(MIN_RELEVANT_TO_WEB_SEARCH)
    )

    graph.add_edge(
        "vector_db",
        "rag",
        out=lambda input, result: ({**input, **result}),
        condition=lambda result: len(result) > 0,
        description="docs > 0"
    )

    graph.add_edge(
        "elasticsearch",
        "web_search",
        out=lambda input, result: ({**input, "documents": input["documents"] + result}),        
    )

    graph.add_edge(
        "web_search",
        "rag",
        out=lambda input, result: ({**input, "documents": result}),
        condition=lambda result: len(result) > 0,
        description="docs > 0"
    )

    graph.add_edge("rag", "stream_results",
                   out=lambda input, result: ({
                       **input,
                       "result": result
                   }))

    return graph


def create_medline_graph():

    graph = Graph()

    setup_graph_memory(graph)

    graph.add_node("medline", elasticsearch_node)
    graph.add_node("generate", generate)
    graph.add_node("stream_results", stream_results)
  

    graph.add_edge(
            "medline",
            "generate",
            out=lambda input, result: ({**input, "documents": result}),
            condition=lambda result: len(result) > 0,
            description="docs > 0"
        )

    graph.add_edge("generate", "stream_results",
                    out=lambda input, result: ({
                        **input,
                        "result": result
                    }))


    return graph
