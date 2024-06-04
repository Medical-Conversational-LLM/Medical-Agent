import numpy as np
import hnswlib
from retrieval_dataset import retrieval_df
from lm.retriever.portable_db_vector import search_in_documents
from lm.retriever.utils import get_model, filter_index_results

INDEX_FILE = "storage/index/medical-records.bin"
EMBEDDINGS_FILE = "storage/index/medical-records.npy"
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
        results, distances = search_in_documents(input["query"], results)
    else:
        distances = []
        results = []

    graph.streamer.put({
        "type": "DB_SEARCH",
        "message": "Found {} articles".format(len(results)),
    })
 
    
    average_score = 1
    if len(results) > 0:
        average_score = np.average(distances[0::5])

    return {"documents": results, "avg": average_score}
