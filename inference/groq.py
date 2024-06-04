from inference.inference import Inference
from groq import Groq as GroqClient
from utils import get_cloud_formatted_prompt

class Groq(Inference):
    client: GroqClient
    model: str

    def __init__(self, api_key: str, model="llama3-8b-8192") -> None:
        super().__init__()
        self.client = GroqClient(api_key=api_key)
        self.model = model

    def completion(self, prompt: str) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=get_cloud_formatted_prompt(prompt),
            model=self.model,
        )

        return chat_completion.choices[0].message.content
