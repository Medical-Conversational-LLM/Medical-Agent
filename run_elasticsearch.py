from lm.retriever.elasticsearch import elasticsearch_search_documents
from utils import normalize_data 


query = normalize_data("cancer")

documents = elasticsearch_search_documents(query)