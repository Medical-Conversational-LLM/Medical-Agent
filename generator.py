# from data_collection.collector import Collector
# from inference.chatgpt import Chatgpt
# from inference.mocked_inference import MockedInference
# from reader.csv import CSV
# from prompt.utility import Utility
# from prompt.relevance import Relevance
# from prompt.retrieval import Retrieval
# from prompt.supported import Supported
# from writer.json import JsonWriter
# from inference.pool import Pool
# from inference.groq import Groq
# from cache.tiny_db import TinyDBCache
# import random
# from writer.writer import Writer

# def retrieve_documents(x):
#       return [{"title": "Document 1", "text": "Content of document 1"}, {"title": "Document 2", "text": "Content of document 2"}]


# def run_critic_model(x,seg):
#     reflection_tokens = {
#     "Retrieve": random.choice(["Yes", "No"]),
#     "Relevant": random.choice(["Relevant", "Not Relevant"]),
#     "Supported": random.choice(["Supported", "Not Supported"]),
#     "Utility": random.choice(["Useful", "Not Useful"])
# }
#     return reflection_tokens


# reader=CSV(
#     "./datasets/updated_PubMedQA_pqa_artificial.csv",
#     skip_header=True,
#     mapper=lambda row: dict(
#         {
#             "id": row[0],
#             "input": row[1],
#             "evidence": row[2],
#             "output": row[3],
#         }
#     ),
# )
# rows = reader.read()

# slice=200
# augmented_data = []
# for data in rows:
#     x= data['input']
#     y=data['output']


#     segments = y.split('.')
#     augmented_segments = []
#     for segment in segments:
#         reflection_tokens = run_critic_model(x, segment)

#         if reflection_tokens['Retrieve'] == 'Yes' :
#             retrieved_docs = retrieve_documents(x)

#             for doc in retrieved_docs:
#                 augmented_segments.append({
#                     "input": x,
#                     "segment": segment,
#                     "retrieved_doc": doc,
#                     "reflection_tokens": reflection_tokens
#                 })
#         else:
#             augmented_segments.append({
#                 "input": x,
#                 "segment": segment,
#                 "reflection_tokens": reflection_tokens
#             })
#     augmented_data.append({"input": x, "output": y, "augmented_segments": augmented_segments})

# writer=JsonWriter("./storage/output/generator_DS.json"),
# writer.append(
#     {
#         augmented_data
#     }
# )