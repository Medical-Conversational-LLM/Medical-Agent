from graph.graph import Graph
from inference.pretrained import Pretrained
from lm.check_need_retrieval import check_need_retrieval
retrieval_inference = Pretrained("HlaH/Llama3-ChatQA-Retriever-PubMedQA")

graph = Graph()
graph.set_memory("retrieval_inference", retrieval_inference)

check_need_retrieval({
    "query": "What is the role of insulin in regulating blood sugar levels?"
}, graph)
