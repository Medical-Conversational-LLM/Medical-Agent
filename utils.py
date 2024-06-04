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
            if item is None:
                break
            yield item

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