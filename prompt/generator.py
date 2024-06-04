from prompt.prompt import Prompt
from typing import Any
from utils import create_training_object, clean_text


PROMPT_DICT = {
    "prompt_input": (
"""
I have a yes/no question that needs to be converted into an open-ended question and get the answer. Here is the information:

Question: {question}
Context: {context}
Long Answer: {long_answer}
Final Decision: {final_decision}
Level of Evidence (LOE): {loe}

Convert the above yes/no question into an open-ended question.Use the provided information/Question to generate a comprehensive answer.
Do not include extra text to the response ,foe example : Here is the converted question and answer: . Provide the response in the following format :
Question:
Answer:
"""
    )
}

def create_result(Response):
    return {"Response": Response}


import ast
class Genertator(Prompt):

    def format(self, data: dict[str, str]) -> str:
        return PROMPT_DICT["prompt_input"].format_map(data)

    def parse(self, result: str, row: Any) -> str:


        return {"token": clean_text(result), 
                 "question": result, "context": row["context"], "loe": row["loe"], "long_answer":row['long_answer'], 
                    "final_decision":row['final_decision'] }
    
    