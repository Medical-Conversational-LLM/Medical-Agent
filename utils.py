import queue
import re
import unicodedata
import json
import json
import random
import string


def clean_text(text):
    text = unicodedata.normalize("NFC", text)

    text = text.encode("utf-8", "ignore").decode("utf-8")

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text


def create_training_object(instruction: str, input: str, label: str, task: str):
    return {
        "instruction": instruction,
        "input": input,
        "output": label,
        "task": task,
    }


def handle_exception(exception: Exception):
    if isinstance(exception, ValueError):
        import traceback

        trace = traceback.extract_tb(exception.__traceback__)[-1]
        print("####### ERROR ######")
        print("\t{}".format(exception))
        print((
            "\tFunction: {name}\n"
            "\tFile: '{filename}\n"
            "\tLine: {end_lineno}, Column: {end_colno}"
        ).format(
            name=trace.name,
            filename=trace.filename,
            end_lineno=trace.end_lineno,
            end_colno=trace.end_colno
        ))
    else:
        raise exception


def match_reflective_token_with_explaination(text: str):
    first, *explanation = text.split("\nExplanation:")

    explanation = "\n".join(explanation)

    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, first)

    return [matches[0] if matches is not None and len(matches) > 0 else None, explanation.strip()]


def merge_explanations(explanations: list[tuple[str, str]]):
    return "\n\n".join(
        [
            "*{title}:*\n{explanation}".format(
                title=explanation[0], explanation=explanation[1]
            ) for explanation in explanations
        ]
    )


def to_stream(message):
    if isinstance(message, dict) or isinstance(message, list):
        message = json.dumps(message)
    return 'data: {message}\n\n' .format(message=message)


def to_stream_response(generator):
    for item in generator: 
        if item is None:
            break  
        yield to_stream(item)


def generate_id(length=16):
    characters = string.ascii_letters + string.digits
    result_str = ''.join(random.choice(characters) for i in range(length))
    return result_str

class ThreadStreamer:
    queue = queue.Queue()

    def put(self, item):
        self.queue.put(item)

    def get(self):
        while True:
            item = self.queue.get()
            yield item
            if item is None:
                break


    def end(self):
        print("FINISHED ..")
        self.queue.put(None)


def should_continue(result, msg=None):
    return "continue" not in result or result["continue"] == True


def get_llama_formatted_prompt(messages):
    return "\n\n".join([
        "{role}: {content}".format(role=message["role"].title(), content=message["content"]) for message in messages
    ])
    

def get_cloud_formatted_prompt(prompt):
    if type(prompt) is list:
        return prompt
    
    return [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]


import string
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def normalize_data(query):
    query = clean_text(query)
    
    query = query.lower()
    
    # Remove punctuation
    query = query.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stop words
    medical_stop_words = set(["a", "an", "and", "the", "but", "or", "on", "in", "with", "without", "of", "at", "by", "are", "is"])
    query = ' '.join(word for word in query.split() if word not in medical_stop_words)
    
    # Tokenize
    query_tokens = word_tokenize(query)
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    query_tokens = [lemmatizer.lemmatize(word) for word in query_tokens]
    
    # Expand with medical synonyms
    def expand_query_with_synonyms(tokens):
        return tokens
        expanded_tokens = []
        for word in tokens:
            synonyms = wordnet.synsets(word)
            expanded_tokens.append(word)
            for syn in synonyms:
                for lemma in syn.lemmas():
                    expanded_tokens.append(lemma.name())
        return list(set(expanded_tokens))  # remove duplicates

    query_tokens = expand_query_with_synonyms(query_tokens)
    
    
    
    return " ".join(query_tokens)


 