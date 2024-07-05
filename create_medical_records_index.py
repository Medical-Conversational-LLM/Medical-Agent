
import os

output_npy = 'storage/index/medical-records.npy'
output_bin = 'storage/index/medical-records.bin'



# Check if the file exists
if not os.path.isfile(output_npy) and not os.path.isfile(output_bin):

    from logger import logger, profiler
    from sentence_transformers import SentenceTransformer
    import hnswlib
    from retrieval_dataset import retrieval_df
    import numpy as np

    name = "medical-record"
    out_name="medical-record" 
    key = "article"

    model = SentenceTransformer('all-MiniLM-L6-v2')

    logger.info("creating embeddings")
    embeddings = model.encode(retrieval_df[key], convert_to_numpy=True, show_progress_bar=True, device="cuda")
    
    logger.info("creating index")

    num_dim = embeddings.shape[1]
    
    logger.info("saving embeddings")
    np.save(output_npy.format(key), embeddings)

    index = hnswlib.Index(space='cosine', dim=num_dim)

    index.init_index(max_elements=embeddings.shape[0], ef_construction=200, M=16)

    logger.info("adding index items")
    index.add_items(embeddings)

    logger.info("saving index")
    index.save_index(output_bin.format(key))


    logger.info("finished")

    profiler.profile("END")

else:
    print("== Medical Records already installed ==")
    print("no need to create medical records")