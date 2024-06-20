from sentence_transformers import SentenceTransformer, CrossEncoder
cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512, device="cuda")
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


COSINE_THRESHOLD = 0.5


def filter_index_results(labels, distances, data):
    final_distances = []
    final_labels = []
    
    for label, distance in zip(labels[0], distances[0]):
        if distance > COSINE_THRESHOLD:
            continue
        if data[label] is None or data[label].strip() == "":
            continue
        final_labels.append(data[label])
        final_distances.append(distance)

    return final_labels, final_distances


def rerank_labels_with_cross_encoder(query, labels):
    if not labels:
        return []

    pairs = [(query, str(chunk)) for chunk in labels]
    if not pairs:
        return []
    
    try:
        scores = cross_encoder.predict(pairs)

        sorted_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        sorted_labels = [labels[i] for i in sorted_indices]

        return sorted_labels
    except Exception as e:
        print(f"An error occurred during cross_encoder.predict: {e}")
        return labels


def get_relevant_summarization(chunk):
    LANGUAGE = "english"
    SENTENCES_COUNT = 3
    arr=[]

    parser = PlaintextParser.from_string(chunk, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summarized_sentences = [str(sentence) for sentence in summarizer(parser.document, SENTENCES_COUNT)]
    
    joined_sentences =  ' '.join(summarized_sentences)

    arr.append(joined_sentences)
    return arr
