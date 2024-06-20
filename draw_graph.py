from lm.create_graph import create_graph, create_rag_graph
from graph.graph import Edge
import graphviz  
from numpy import isscalar

graph = create_rag_graph()

dot = graphviz.Digraph(comment='Self-Rag Graph')

def get_edge_description(edge: Edge):
    if  edge["description"]  is not None:
        return edge["description"]   
    
    if(edge["condition"] is not None and isscalar(edge["condition"])):
        return str(edge["condition"])
    
for node in graph.nodes:
    dot.node(node["name"])     

for edge in graph.edges:
    dot.edge(edge["from_node"], edge["to_node"],get_edge_description(edge))
    

print("PASTE OUTPUT INTO -> https://dreampuf.github.io/GraphvizOnline")
print(dot.source) 