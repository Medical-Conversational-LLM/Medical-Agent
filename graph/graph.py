import types
from typing import TypedDict, Callable, Any
import inspect
import time


class Node(TypedDict):
    name: str
    callback: Callable[[], None]


class Edge(TypedDict):
    from_node: str
    to_node: str
    condition: Any
    out: Any
    description: str


class NullStreamer:
    def put(self, item):
        print(item)        

    def get(self):
        pass

    def end(self):
        pass


class Graph:
    nodes: list[Node]
    edges: list[Edge]

    memory: dict[str, str]
    streamer: NullStreamer

    def __init__(self) -> types.NoneType:
        self.nodes = list()
        self.edges = list()

        self.memory = dict()
        self.streamer = NullStreamer()

    def add_node(self, name: str, callback: Callable[[], None]):
        self.nodes.append({
            "name": name,
            "callback": callback
        })

        return self

    def add_edge(self, from_node, to_node, condition: Any = None, out=None, description=None):
        self.edges.append({
            "from_node": from_node,
            "to_node": to_node,
            "condition": condition,
            "out": out,
            "description": description
        })
        return self

    def set_memory(self, key: str, value: str):
        self.memory[key] = value

    def get_memory(self, key: str, default=None):
        value = self.memory.get(key)
        return value if value is not None else default

    def find_node_by_name(self, name: str):
        for node in self.nodes:
            if node["name"] == name:
                return node

    def find_outgoing_edges(self, node_name: str):
        return [edge for edge in self.edges if edge["from_node"] == node_name]

    def run_node(self, node: Callable[[], None], input: dict[str, str]):
        return call_fn(node["callback"], input, self)

    def start(self, name: str, input: dict[str, str] = {}):

        self.run(name, input)

        time.sleep(0.1)
        self.streamer.end()

    def run(self, name: str, input: dict[str, str] = {}):
        node = self.find_node_by_name(name)
        if node is None:
            raise ValueError("node [{}] not found".format(name))
        self.current_node_name = name
        print("running {}".format(name))

        result = self.run_node(node, input)
 

        for edge in self.find_outgoing_edges(name):
            can_run = True

            if edge["condition"] != None:
                if is_callable(edge["condition"]):
                    can_run = call_fn(
                        edge["condition"], result, input, self)
                else:
                    can_run = edge["condition"] == result

            if can_run == True:
                input = result if edge["out"] is None else call_fn(
                    edge["out"], input, result, self)

                self.run(edge["to_node"], input)

        return self


def call_fn(fn, p1=None, p2=None, p3=None):
    sig = inspect.signature(fn)
    num_params = len(sig.parameters)
    if num_params == 0:
        return fn()
    elif num_params == 1:
        return fn(p1)
    elif num_params == 2:
        return fn(p1, p2)
    else:
        return fn(p1, p2, p3)


def is_callable(value):
    return (
        isinstance(value, types.LambdaType) and value.__name__ == "<lambda>"
    ) or (isinstance(value, types.FunctionType))
