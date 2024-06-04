from inference.pretrained import Pretrained
from prompt.supported import SupportedToken
from graph.graph import Graph
from lm.generate import generate

graph = Graph()
inference = Pretrained("HlaH/Llama3-ChatQA-Generator-PubMedQA")
graph.set_memory("generator_inference", inference)
graph.add_node("generate", generate)

graph.run("generate", {
    "query": "What is caner",
    "documents": [
        """
Cancer is a disease of the body's cells. Normally cells grow and multiply in a controlled way, however, sometimes cells become abnormal and keep growing. Abnormal cells can form a mass called a tumour. Cancer is the term used to describe collections of these cells, growing and potentially spreading within the body.

"""
    ]
})
