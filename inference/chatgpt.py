from inference.inference import Inference
from openai import OpenAI
from utils import get_cloud_formatted_prompt

class Chatgpt(Inference):
    client: OpenAI

    def __init__(self, api_key: str) -> None:
        super().__init__()

        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=api_key
        )

    def completion(self, prompt: str) -> str:
        
        chat_completion = self.client.chat.completions.create(
            messages=get_cloud_formatted_prompt(prompt),
            model="gpt-3.5-turbo",
            # request_timeout=60,
            # max_tokens=200,
        )

        return chat_completion.choices[0].message.content
