from lm.create_graph import create_graph
from logger import profiler
from utils import handle_exception, list_available_devices, normalize_data
import torch




list_available_devices()

while True:
    try:
        query = input("Enter your question: ")
        break
    except Exception as error:
        handle_exception(error)


graph = create_graph()
result = graph.start("check_retrieval", {
    "query": normalize_data(query),
    "chat_history": []
})


# # profiler.profile("DONE TOOK")
# import torch

# # For PyTorch
# with torch.cuda.device(1):  # Adjust device number as needed
#     torch.cuda.empty_cache()
