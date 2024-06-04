from prompt.prompt import Prompt
from typing import Any
from utils import create_training_object, clean_text

MULTI_INSTRUCTION=  ("You'll be given an input along with evidence/retrieved docs, and possibly some preceding sentences. ")

multi_no_preceding=  ( "You'll be provided with an input, along with evidence and possibly some preceding sentences. ")
PROMPT_DICT = {
   
    "multi": (
        "instruction:You'll be given an instruction along with evidence/retrieved docs, and possibly some preceding sentences. "
         "When there are preceding sentences, focus on the sentence that comes after them. "
        "Your task is to determine if the evidence is relevant to the initial instruction and the preceding context, and provides useful information to complete the task described in the instruction. "
        "If the evidence meets this requirement, respond with [Relevant]; otherwise, generate [Irrelevant].\n\n"
        "input: Choose the correct diagnosis based on a patient's symptoms.\n\n"
        "Symptoms: fever, cough, fatigue\n"
        "A: Influenza\nB: Common cold\nC: Pneumonia\nD: Asthma\n\n"
        "evidence: The patient presents with fever, cough, and fatigue, which are common symptoms of both influenza and pneumonia.\n\n"
        "output: [Relevant]\n"
        "###\nInstruction: Explain the function of the hippocampus in memory formation.\n\n"
        "evidence: Dr. Emily Johnson is a leading neuroscientist known for her groundbreaking research on memory formation. She has published extensively on the role of the hippocampus and its interactions with other brain regions.\n\n"
        "output: [Irrelevant]\n"
        "###\n input: {input}\n\n"
        "evidence: {evidence}\n\n"
        "output:"
    ),
    "multi_no_preceding": (
      "You'll be provided with an input, along with evidence and possibly some preceding sentences.When there are preceding sentences, your focus should be on the sentence that comes after them.Your job is to determine if the evidence is relevant to the initial instruction and provides useful information to complete the task described. If the evidence meets this requirement, respond with [Relevant]; otherwise, generate [Irrelevant].\n\n"
        "Input:  Describe a common complication of diabetes.\n\n"
        "Preceding sentences: Diabetes is a chronic condition characterized by high levels of sugar (glucose) in the blood. It can lead to various health complications if not managed properly.\n\n"
        "Evidence: Dr. John Smith is a renowned endocrinologist who specializes in the treatment of diabetes. He has published numerous research papers on the topic and has received several awards for his contributions to the field.\n\n"
        "Response: [Irrelevant]\n"
        "###\n Input: Describe the symptoms of asthma.\n\n"
        "Evidence: Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, leading to symptoms such as wheezing, shortness of breath, chest tightness, and coughing.\n\n"
        "Response: [Relevant]\n"
        "###\nInput: Explain the mechanism of action of statins.\n\n"
        "Evidence: Statins work by inhibiting the enzyme HMG-CoA reductase, which plays a key role in cholesterol synthesis. By reducing cholesterol production, statins help lower blood cholesterol levels and reduce the risk of cardiovascular disease.\n\n"
        "Response: [Relevant]\n"
        "###\nInput: {input}\n\n"
        "Evidence: {evidence}\n\n"
        "Response:"
    ),
}


def create_result(output):
    return {"output": output}


OUTPUT_INSTRUCTIONS = (
    "When given input and evidence, evaluate whether the evidence is relevant to the instruction and provides valuable information for generating meaningful responses.Use a rating of [Relevant] to indicate relevance and usefulness, and [Irrelevant] to indicate irrelevance."
)

import ast
class Relevance(Prompt):

    def format(self, data: dict[str, str]) -> str:
        return PROMPT_DICT["multi"].format_map(data)

    def parse(self, result: str, row: Any) -> str:

        data = ast.literal_eval(row["evidence"])
        if 'contexts' in data:
            row["evidence"]= data['contexts']

        return {"token": clean_text(result), "instruction":multi_no_preceding,"task" :'relevance',
                 "input": row["input"] + " \nEvidence:" + row["evidence"][0]}
    
        return create_training_object(
            OUTPUT_INSTRUCTIONS,
            ("Task instruction: {instruction}\n" "Evidence: {evidence}").format_map(
                row
            ),
            clean_text(rating),
            "relevance",
        )
