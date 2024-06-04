from sentence_transformers import SentenceTransformer


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
