from elasticsearch import Elasticsearch

from graph.graph import Graph

elastic_url = 'http://ariadne.is.inf.uni-due.de:7777'
username = 'elastic'
password = 'ppiiPPhh22iieell55aaaatt'


es = Elasticsearch(elastic_url, http_auth=(
    username, password))


index_name = 'medline'
return_columns = ['pmid', 'title', 'abstract', 'citations', 'loe', 'authors']
search_columns = ['title', 'abstract']


def elasticsearch_search_documents(query: str):
    try:
        search_query = {
            "query": {
                "multi_match": {
                    "fields": search_columns,
                    "query": query,
                    "fuzziness": "AUTO"
                }},
            "size": 20,
        }

        response = es.search(index=index_name,  body=search_query)

        return ["{abstract} PUBMIDID({pmid}) LOE({loe}) AUTHORS({authors})".format(
            abstract=item["_source"]["abstract"],
            pmid=item["_source"]["pmid"],
            loe=item["_source"]["loe"] if "loe" in item["_source"] else None,
            authors=",".join(item["_source"]["authors"]
                             ) if "authors" in item["_source"] else "",

        ) for item in response["hits"]["hits"]]

    except:
        print("failed to call elasticsearch")
        return []


def elasticsearch_node(input: str, graph: Graph):
    graph.streamer.put({
        "type": "ELASTIC_SEARCH",
        "message": "Elasticsearch lookup"
    })

    return elasticsearch_search_documents(input["query"])
