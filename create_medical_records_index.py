from sentence_transformers import SentenceTransformer
import hnswlib
from retrieval_dataset import retrieval_df
import numpy as np

name = "medical-record"
out_name="medical-record"
# key = "article"
key = "abstract"

model = SentenceTransformer('all-MiniLM-L6-v2')

print("creating embeddings")
embeddings = model.encode(retrieval_df[key], convert_to_numpy=True, show_progress_bar=True, device="cuda")
 
print("creating index")

num_dim = embeddings.shape[1]
 
print("saving embeddings")
np.save('storage/index/medical-records.npy'.format(key), embeddings)

index = hnswlib.Index(space='cosine', dim=num_dim)

index.init_index(max_elements=embeddings.shape[0], ef_construction=200, M=16)

print("adding index items")
index.add_items(embeddings)

print("saving index")
index.save_index('storage/index/medical-records.bin'.format(key))


print("finished")