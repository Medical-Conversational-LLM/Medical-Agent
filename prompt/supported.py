import ast
from prompt.prompt import Prompt
from typing import Any
from utils import create_training_object, clean_text
import json
from enum import Enum


# class syntax
class SupportedToken(Enum):
    FULL = "Fully supported"
    PARTIAL = "Partially supported"
    NO_SUPPORT = "No support"


SUPPORTED_EXPLAINATION_SUMMARY_TITLE = (
    "evaluation if the output is fully supported by the information"
    "provided in the evidence provided previously"
)

INSTRUCTION = (
    "You will receive an input, evidence, and output.\n"
    "Your task is to evaluate if the output is fully supported by the information provided in the evidence.\n"
    "Use the following entailment scale to generate a score:\n"
    "5: Fully supported - All information in output is supported by the evidence, or extractions from the evidence. This is a somewhat extreme case and is only applicable when the output and part of the evidence are almost identical.\n"
    "4: Mostly supported - Most of the information in the output is supported by the evidence, but there is some minor information that is not supported. In other words, if an output is a paraphrase of the evidence or a less concrete version of the descriptions of the evidence, it should be considered a 4.\n"
    "3: Partially supported - The output is supported by the evidence to some extent, but there is major information in the output that is not discussed in the evidence. For example, if an instruction asks about two concepts and the evidence only discusses either of them, it should be considered a 3. If the output covers a lot of new information that is not discussed in the evidence, it should be 3.\n"
    "2: Little support - The output and evidence are only loosely related, and most of the information in the output isn't supported by the evidence.\n"
    "1: Ignore / Contradictory - The output completely ignores evidence or contradicts the evidence. This can also happen if the evidence is irrelevant to the instruction.\n"
)

INSTRUCTION_multi = ("You will receive an input, evidence, and output, and optional preceding sentences.  If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to evaluate if the output is fully supported by the information provided in the evidence, and provide explanations on your judgement\n")
INSTRUCTION_multi_no_preceding = ("You will receive an instruction, evidence, and output, and optional preceding sentences.  If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to evaluate if the output is fully supported by the information provided in the evidence, and provide explanations on your judgement.\n")

PROMPT_DICT = {
    "context": (
        "instruction:You will receive an input, evidence, and output.\n"
        "Your task is to evaluate if the output is fully supported by the information provided in the evidence.\n"
        "Use the following entailment scale to generate a score:\n"
        "5: Fully supported - All information in output is supported by the evidence, or extractions from the evidence. This is a somewhat extreme case and is only applicable when the output and part of the evidence are almost identical.\n"
        "4: Mostly supported - Most of the information in the output is supported by the evidence, but there is some minor information that is not supported. In other words, if an output is a paraphrase of the evidence or a less concrete version of the descriptions of the evidence, it should be considered a 4.\n"
        "3: Partially supported - The output is supported by the evidence to some extent, but there is major information in the output that is not discussed in the evidence. For example, if an instruction asks about two concepts and the evidence only discusses either of them, it should be considered a 3. If the output covers a lot of new information that is not discussed in the evidence, it should be 3.\n"
        "2: Little support - The output and evidence are only loosely related, and most of the information in the output isn't supported by the evidence.\n"
        "1: Ignore / Contradictory - The output completely ignores evidence or contradicts the evidence. This can also happen if the evidence is irrelevant to the instruction.\n"
        "Make sure to not use any external information/knowledge to judge whether the output is true or not.\n"
        "Only check whether the output is supported by the evidence, and not whether the output follows the instructions or not.\n\n"
        "###\n Input: Describe the symptoms of COVID-19.\n\n Output: Common symptoms of COVID-19 include fever, dry cough, and fatigue. Some patients may also experience body aches, loss of taste or smell, sore throat, and difficulty breathing.\n\n"
        "Evidence: COVID-19 symptoms\nCommon symptoms of COVID-19 include fever, dry cough, and fatigue. Some patients may also experience body aches, loss of taste or smell, sore throat, and difficulty breathing.\n\n"
        "###Response: Fully supported\n"
        "\n\nExplanation: The output perfectly matches the information provided in the evidence regarding the symptoms of COVID-19.\n\n"
        "###\n Input: Identify the risk factors for heart disease.\n\n Output: Risk factors for heart disease include high blood pressure, high cholesterol, smoking, obesity, diabetes, and family history of heart disease.\n\n"
        "Evidence: Risk factors for heart disease\nCommon risk factors for heart disease include high blood pressure, high cholesterol, smoking, obesity, diabetes, and family history of heart disease.\n\n"
        "###Response: Fully supported\n"
        "\n\nExplanation: The output directly corresponds to the risk factors mentioned in the evidence for heart disease.\n\n"
        "###\n Input: {input}\n\n , \n Output:{Output}\n" "Evidence: {evidence}\n\n"
        "###Response:\n"
        "\n\nExplanation:\n\n"
    ),
    "multi": (
        "instruction:You will receive an input, evidence, and output, and optional preceding sentences.  If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to evaluate if the output is fully supported by the information provided in the evidence, and provide explanations on your judgement\n"
        "Use the following entailment scale to generate a score:\n"
        "[Fully supported] - All information in output is supported by the evidence, or extractions from the evidence. This is only applicable when the output and part of the evidence are almost identical.\n"
        "[Partially supported] - The output is supported by the evidence to some extent, but there is major information in the output that is not discussed in the evidence. For example, if an instruction asks about two concepts and the evidence only discusses either of them, it should be considered a [Partially supported].\n"
        "[No support / Contradictory] - The output completely ignores evidence, is unrelated to the evidence, or contradicts the evidence. This can also happen if the evidence is irrelevant to the instruction.\n\n"
        "Make sure to not use any external information/knowledge to judge whether the output is true or not. Only check whether the output is supported by the evidence, and not whether the output follows the instructions or not.\n\n"
        "###\n Input: Describe the symptoms of COVID-19.\n\n Output: Common symptoms of COVID-19 include fever, dry cough, and fatigue. Some patients may also experience body aches, loss of taste or smell, sore throat, and difficulty breathing.\n"
        "Evidence: COVID-19 symptoms\nCommon symptoms of COVID-19 include fever, dry cough, and fatigue. Some patients may also experience body aches, loss of taste or smell, sore throat, and difficulty breathing.\n"
        "###Response: [Fully supported]\n"
        "\n\nExplanation: The output perfectly matches the information provided in the evidence regarding the symptoms of COVID-19.\n\n"
        "###\n Input: Identify the risk factors for heart disease.\n\n Output: Risk factors for heart disease include high blood pressure, high cholesterol, smoking, obesity, diabetes, and family history of heart disease.\n"
        "Evidence: Risk factors for heart disease\nCommon risk factors for heart disease include high blood pressure, high cholesterol, smoking, obesity, diabetes, and family history of heart disease.\n"
        "###Response: [Fully supported]\n"
        "\n\nExplanation: The output directly corresponds to the risk factors mentioned in the evidence for heart disease.\n\n"
        "###\n Input: {input}\n , \n Output:{Output}\n evidence: {evidence}\n"
        "Preceding sentences: {preceding_sentences}\n"
        "###Response:\n , Explanation: \n\n"


    ),
    "multi_no_preceding": (
        "instruction:You will receive an input, evidence, and output, and optional preceding sentences.\n"
        "If the preceding sentence is given, the output should be the sentence that follows those preceding sentences.\n"
        "Your task is to evaluate if the output is fully supported by the information provided in the evidence.\n"
        "Please provide short explanation on your judgement max (10 words)\n"
        "Use the following entailment scale to generate a score:\n"
        "[Fully supported] - All information in output is supported by the evidence, or extractions from the evidence. This is only applicable when the output and part of the evidence are almost identical.\n"
        "[Partially supported] - The output is supported by the evidence to some extent, but there is major information in the output that is not discussed in the evidence. For example, if an instruction asks about two concepts and the evidence only discusses either of them, it should be considered a [Partially supported].\n"
        "[No support / Contradictory] - The output completely ignores evidence, is unrelated to the evidence, or contradicts the evidence. This can also happen if the evidence is irrelevant to the instruction.\n\n"
        "Make sure to not use any external information/knowledge to judge whether the output is true or not. Only check whether the output is supported by the evidence, and not whether the output follows the instructions or not.\n\n"
        "Example 1:\n"
        "Input: Describe the symptoms of COVID-19.\n"
        "Output: Some common symptoms of COVID-19 include fever, dry cough, and fatigue. In severe cases, patients may also experience difficulty breathing and chest pain.\n"
        "Evidence: COVID-19 symptoms\nCommon symptoms of COVID-19 include fever, dry cough, and fatigue. Some patients may also experience body aches, loss of taste or smell, sore throat, and difficulty breathing.\n"
        "Response: [Partially supported]\n"
        "\n\nExplanation: The output describes additional symptoms of COVID-19 not mentioned in the evidence, such as difficulty breathing and chest pain.\n\n"
        "USER INPUT:\n\n"
        "Input: {input}\n"
        "\n Output:{Output}\n"
        "Evidence: {evidence}\n"
        "Response: "
        "\n\nExplanation: ")
}


def create_result(response, explanation: str):
    return {"response": response, "explanation": explanation}


OUTPUT_INSTRUCTIONS = (
    "You will be given an task Input, Evidence, and Output. Your objective is to assess the extent to which the output is supported by the information presented in the evidence.\n"
    "Rate the level of support on a scale from 1 ( Ignore / Contradictory), 2 (Little support), 3 (Partially supported), 4 (Mostly supported), 5 (Fully supported)."
)


class Supported(Prompt):

    def format(self, data: dict[str, str]) -> str:
        return PROMPT_DICT["multi_no_preceding"].format_map(data)

    def parse(self, result: str, row: Any) -> str:

        data = ast.literal_eval(row["evidence"])
        if 'contexts' in data:
            row["evidence"] = data['contexts']

        return {
            "instruction": INSTRUCTION_multi_no_preceding,
            "token": clean_text(result),
            "task": 'groudness',
            "input": row["input"] + " \nOutput:" + row["Output"] + " \nEvidence:" + row["evidence"][0]
        }

        return create_training_object(
            OUTPUT_INSTRUCTIONS,
            (
                "##\nTask instruction: {instruction}\n"
                "Evidence: {evidence}\n"
                "Output: {output}"
            ).format_map(row),
            clean_text(rating),
            "supported",
        )
