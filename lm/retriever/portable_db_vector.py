from semantic_text_splitter import TextSplitter
import hnswlib
import numpy as np
from lm.retriever.utils import get_model, filter_index_results, rerank_labels_with_cross_encoder

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

def search_in_documents(query: str, documents: list[str]):
    splitter = TextSplitter((1600))
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

    results = filter_index_results(labels, distances, documents)

    return results
    # return rerank_labels_with_cross_encoder(query, labels[0])


def get_relevant_summarization(query: str, documents: list[str]):
    LANGUAGE = "english"
    SENTENCES_COUNT = 5
    full_documents = documents
    documents = []

    for document in full_documents:
        arr=[]
        # for index, chunk in enumerate(chunks):
        parser = PlaintextParser.from_string(document, Tokenizer(LANGUAGE))
        stemmer = Stemmer(LANGUAGE)

        summarizer = Summarizer(stemmer)
        summarizer.stop_words = get_stop_words(LANGUAGE)

        summarized_sentences = [str(sentence) for sentence in summarizer(parser.document, SENTENCES_COUNT)]
        
        # Join all sentences into a single string with "loe" and collect the results
        joined_sentences =  ' '.join(summarized_sentences)

        arr.append(joined_sentences)
        return arr
