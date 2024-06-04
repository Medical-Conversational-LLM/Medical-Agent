from graph.graph import Graph
from lm.check_need_retrieval import check_need_retrieval
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
from utils import should_continue
inference = Groq(
    api_key="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",
    model="llama3-8b-8192",
)
critique_inference = Pretrained("HlaH/Llama3-ChatQA-Critic-PubMedQA")
retrieval_inference = Pretrained("HlaH/Llama3-ChatQA-Retriever-PubMedQA")
generator_inference = Pretrained("HlaH/Llama3-ChatQA-Generator-PubMedQA")
MIN_RELEVANT_TO_WEB_SEARCH = 7
MIN_AVG_TO_WEB_SEARCH = 0.3


# summarization_inference = inference


def create_graph():

    graph = Graph()

    graph.set_memory("retrieval_inference", retrieval_inference)
    graph.set_memory("relevance_inference", critique_inference)
    graph.set_memory("critique_supported_inference", critique_inference)
    graph.set_memory("critique_utility_inference", critique_inference)
    graph.set_memory("generator_inference", critique_inference)
    
    # graph.set_memory("summarization_inference", summarization_inference)

    graph.add_node("check_retrieval", check_need_retrieval)
    graph.add_node("vector_db", vector_db)
    graph.add_node("web_search", web_search)
    graph.add_node("critique_relevant", critique_relevant)
    graph.add_node("critique_supported", critique_supported)
    graph.add_node("critique_utility", critique_utility)
    graph.add_node("stream_results", stream_results)
    graph.add_node("generate", generate)
    # graph.add_node("summarize_explanation", summarize_explanation)
    graph.add_node("check_attempts_utility", check_attempts)
    graph.add_node("check_attempts_supported", check_attempts)

    graph.add_edge(
        "check_retrieval",
        "vector_db",
        condition=True,
        out=lambda input: input
    )
    graph.add_edge("check_retrieval", "generate", condition=False)

    graph.add_edge(
        "vector_db",
        "web_search",
        condition=lambda result: len(result['documents']) < MIN_RELEVANT_TO_WEB_SEARCH or
        result['avg'] > MIN_AVG_TO_WEB_SEARCH,
        out=lambda input, result: ({**input, **result}),
        description="Docs < {}".format(MIN_RELEVANT_TO_WEB_SEARCH)
    )

    graph.add_edge(
        "vector_db",
        "critique_relevant",
        condition=lambda result: len(
            result) >= MIN_RELEVANT_TO_WEB_SEARCH and result['avg'] <= MIN_AVG_TO_WEB_SEARCH,
        description="Docs >= {}".format(MIN_RELEVANT_TO_WEB_SEARCH),
        out=lambda input, result: ({**input, **result}),
    )

    graph.add_edge(
        "web_search",
        "critique_relevant",
        out=lambda input, result: ({
            **input,
            "documents": result
        })
    )

    graph.add_edge(
        "critique_relevant",
        "generate",
        out=lambda input, result: ({**input, "documents": result}),
        condition=lambda result: len(result) > 0,
        description="docs > 0"
    )

    graph.add_edge(
        "critique_relevant",
        "check_retrieval",
        out=lambda input, result: input,
        condition=lambda result: len(result) == 0,
        description="docs == 0"
    )

    graph.add_edge("generate", "critique_supported", out=lambda input, result: ({
        **input,
        "result": result,
        "documents": input["documents"]
    }))

    graph.add_edge("critique_supported",
                   "check_attempts_supported",
                   out=lambda input: ({
                       **input,
                   }),
                   condition=lambda result: result["token"] == SupportedToken.NO_SUPPORT,
                   description=SupportedToken.NO_SUPPORT.value
                   )

    graph.add_edge("check_attempts_supported", "check_retrieval", False)

    graph.add_edge("critique_supported",
                   "critique_utility",
                   out=lambda input, result: ({
                       **input,
                       **result
                   }),
                   condition=lambda result: result["token"] !=
                   SupportedToken.NO_SUPPORT,
                   description="SUPPORTED"
                   )

    graph.add_edge("critique_utility",
                   "check_attempts_utility",
                   out=lambda input, result: ({
                       **input,
                       **result,
                       "explanations":  "\n\n".join([
                           result["explanation"],
                           input["explanation"]
                       ])
                   }),
                   condition=lambda result, input: result["score"] < 4 and input['token'] != SupportedToken.FULL,
                   description="utility < 4 and not fully supported")

    graph.add_edge("check_attempts_utility", "check_retrieval",
                   condition=should_continue)

    # graph.add_edge("summarize_explanation",
    #                "check_retrieval",
    #                out=lambda input, result: ({
    #                    **input,
    #                    **result
    #                }),
    #                )

    graph.add_edge("check_attempts_utility", "stream_results",
                   condition=lambda result: should_continue(result, "check_attempts_utility -> stream_results") == False)

    graph.add_edge("critique_utility",
                   "stream_results",
                   out=lambda input, result: ({
                       **input,
                       **result
                   }),
                   condition=lambda result, input: result["score"] >= 4 and input['token'] != SupportedToken.NO_SUPPORT,
                   description="utility = 4 and full or partially supported"
                   )

    return graph


if __name__ == "__main__":
    graph = create_graph()
    result = graph.start("check_retrieval", {
        "query": "what is cancer"
    })
