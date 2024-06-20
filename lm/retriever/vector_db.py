import numpy as np
import hnswlib
from retrieval_dataset import retrieval_df
from lm.retriever.portable_db_vector import search_in_documents , get_relevant_summarization
from lm.retriever.utils import get_model, filter_index_results, rerank_labels_with_cross_encoder

INDEX_FILE = "storage/index/medical-records.bin"
EMBEDDINGS_FILE = "storage/index/medical-records.npy"
# EMBEDDINGS_FILE = "storage/index/embeddings.npy"
M = 16
efC = 100


def load_index():

    embeddings = np.load(EMBEDDINGS_FILE)
    index = hnswlib.Index(space='cosine', dim=embeddings.shape[1])
    index.load_index(INDEX_FILE)
    index.set_ef(200)

    return index


def create_query_embedding(query):

    model = get_model()

    embedding = model.encode([query], normalize_embeddings=True)[0]
    query_embedding_reshaped = embedding.reshape(1, -1)

    return query_embedding_reshaped


def run_query_against_index(query, index, data):
    query_embedding = create_query_embedding(query)

    labels, distances = index.knn_query(query_embedding, 3)
    return filter_index_results(labels, distances, data)

def vector_db(input, graph):

    index = load_index()

    graph.streamer.put({
        "type": "DB_SEARCH",
        "message": "Searching the Database",
    })

    results = run_query_against_index(
        input["query"], index, retrieval_df['article'].iloc)[0]

    if len(results) > 0:
        results =get_relevant_summarization(input["query"], results)
    else:
        distances = []
        results = []
 

    graph.streamer.put({
        "type": "DB_SEARCH",
        "message": "Found {} articles".format(len(results)),
    })
 
    return {"documents": results}
