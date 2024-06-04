from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from retrieval_dataset import retrieval_dataset

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# from lm.retriever.vector_db import load_index,embeddings 

text_data = "WHAT IS CANCER"
model = SentenceTransformer('all-MiniLM-L6-v2')
 
embeddings = model.encode([text_data], convert_to_numpy=True)
 
# index = load_index()
# neighbor_labels, distances = index.knn_query(embeddings, k=3)

# neighbor_vectors = np.array([index.get_items([label]) for label in neighbor_labels[0]])


# all_vectors = np.vstack([embeddings, neighbor_vectors.squeeze()])

