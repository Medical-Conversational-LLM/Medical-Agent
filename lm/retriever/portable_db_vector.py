from semantic_text_splitter import TextSplitter
import hnswlib
import numpy as np
from lm.retriever.utils import get_model, filter_index_results


def search_in_documents(query: str, documents: list[str]):
    splitter = TextSplitter((800))
    full_documents = documents
    documents = []

    for document in full_documents:
        documents.extend(splitter.chunks(document))
        
    model = get_model()

    embeddings = model.encode(documents)

    num_dim = embeddings.shape[1]
    index = hnswlib.Index(space='cosine', dim=num_dim)
    index.init_index(max_elements=len(documents), ef_construction=1000, M=16)
    index.add_items(embeddings, np.arange(len(documents)))

    query_embedding = model.encode([query])

    labels, distances = index.knn_query(query_embedding, k=len(documents))

    return filter_index_results(labels, distances, documents)
