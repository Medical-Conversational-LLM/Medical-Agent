from googleapiclient.discovery import build
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse
from bs4 import Comment
from lm.retriever.portable_db_vector import search_in_documents,get_relevant_summarization
from graph.graph import Graph

GOOGLE_API_KEY = "AIzaSyAgnkSb35XUmfTAtS6CiJ1qW0G3rEDrXsA"
GOOLE_CUSTOM_SEARCH_ENGINE_ID = "9610070cd00304838"


def is_relative_url(url):
    parsed_url = urlparse(url)
    return not parsed_url.scheme and not parsed_url.netloc


def extract_webpage_content(content: str, base_url: str):
    soup = BeautifulSoup(content, 'html.parser')

    tags_whitelist = ["html", "body", "article", "h1", "h2", "h3", "h4", "h5", "6", "span",
                      "main", "details", "figcaption", "mark", "section", "summary", "p", "li", "div", "a", None]

    content = "".join([get_html_element(element, tags_whitelist, base_url)
                      for element in soup.children])

    return content+base_url


def should_add_link_to_extracted_result(link: str):
    if link.startswith("#") is False:
        return False

    if link.startswith("mailto:") is False:
        return False

    return True


def format_link(link: str, base_url: str):
    if is_relative_url(link):
        link = "{base_url}{link}".format(base_url=base_url, link=link)

    return link


def get_html_element(element, tags_whitelist: list[str], base_url: str):

    text = ""

    if element is None or element.name not in tags_whitelist or isinstance(element, Comment):
        return text

    if element.name is None:
        text += element.strip()
    elif element.name == "a":
        text += element.get_text().strip()
        if (href := element.get("href")):
            if should_add_link_to_extracted_result(href) is True:
                text += "{}".format(format_link(href, base_url))

    else:
        for child in element.children:
            text += get_html_element(child, tags_whitelist, base_url)

    return text


def google_search(search_term, num=5, **kwargs):
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    res = service.cse().list(q=search_term, cx=GOOLE_CUSTOM_SEARCH_ENGINE_ID,
                             num=num,  **kwargs).execute()
    return res['items']


def fetch_links_content(links: list[str]):

    results = []

    for link in links:
        try:
            req = Request(link)
            req.add_header('Referer', link)
            req.add_header(
                'User-Agent', "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0")

            response = urlopen(req)
            content = response.read()

            parsed_url = urlparse(link)
            base_url = urlunparse(
                (parsed_url.scheme, parsed_url.netloc, '', '', '', ''))

            results.append(extract_webpage_content(content, base_url))
        except Exception as e:
            print('error=' ,  e)
            continue

    return results


def get_input_documents(input):
        if "documents" in input:
            return input["documents"]
        return []

def web_search(input, graph: Graph):
    if graph.get_memory("web_search") == False:
        return get_input_documents(input)
    
    try:
        results = google_search(input["query"])
        graph.streamer.put({
            "type": "WEB_SEARCH",
            "message": "Searching the web"
        })
        
        initial_links = [result['link'] for result in results]
        
        documents = fetch_links_content(initial_links)

        results_db = get_input_documents(input)
        

        
        additional_links = [None for result in results_db]
        
        all_links = initial_links + additional_links
        
        documents = documents + results_db
    
        documents = search_in_documents(input["query"], documents,all_links)[0][0::6]

        print("documents")
        print(documents)

        graph.streamer.put({
            "type": "WEB_SEARCH",
            "message": "Found {} articles on the web".format(len(documents))
        })
    except:
        graph.set_memory("web_search", False)
        documents = get_input_documents(input)

    return documents
