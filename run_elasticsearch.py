from lm.retriever.elasticsearch import elasticsearch_search_documents
from utils import normalize_data
from lm.retriever.portable_db_vector import search_in_documents


query = normalize_data("cancer")

documents = elasticsearch_search_documents(query)