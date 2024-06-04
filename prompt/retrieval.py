from prompt.prompt import Prompt
from typing import Any
from utils import create_training_object, clean_text

INSTRUCTION = ("You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. If the output sentence can be verified solely with the evidence or doesn’t require any verification, respond with [No Retrieval]. If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments.\n\n")
MULTI_INSTRUCTION = "You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. If the output sentence can be verified solely with the evidence or doesn’t require any verification, respond with [No Retrieval]. If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments.\n\n"
retrieval = "When provided with instruction, please evaluate whether seeking additional information from external sources such as the web (e.g., Wikipedia) aids in producing a more comprehensive response. Respond with either [Retrieval] or [No Retrieval]."

multi_retrieval_no_preceding = "You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. If the output sentence can be verified solely with the evidence or doesn’t require any verification, respond with [No Retrieval]. If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments.\n\n"

multi_retrieval_three_way = "You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. There are three cases:\n"

multi_retrieval_three_way_no_preceding = "You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. There are three cases:\n"

PROMPT_DICT = {

    "context": (
        "instruction:You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. If the output sentence can be verified solely with the evidence or doesn’t require any verification, respond with [No Retrieval]. If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments.\n\n"
        "List three common symptoms of diabetes.\n"
        "Need retrieval?: [Yes]\n"
        "Explanation: There might be some online sources listing common symptoms of diabetes or some reliable sources to explain the medical aspects. So retrieving documents is helpful to improve the response to this query.\n\n"
        "##\nInstruction: Describe a time when you had to explain a complex medical procedure to a patient.\n"
        "Need retrieval?: [No]\n"
        "Explanation: This instruction is asking about some personal experience and thus it does not require one to find some external documents.\n\n"
        "##\nInstruction: Write a short story in third person narration about a protagonist who has to decide whether to undergo a risky surgery.\n"
        "Need retrieval?: [No]\n"
        "Explanation: This instruction asks us to write a short story, which does not require external evidence to verify.\n\n"
        "##\nInstruction: What are the side effects of taking aspirin?\n"
        "Need retrieval?: [Yes]\n"
        "Explanation: While the instruction asks for widely known information, retrieving documents from reliable medical sources can provide a detailed and accurate response.\n\n"
        "##\nInstruction: Calculate the BMI given weight = 70 kg and height = 1.75 m.\n"
        "Need retrieval?: [No]\n"
        "Explanation: This is a simple calculation question and although we may be able to find some documents describing the formula, it is unlikely to find a document exactly mentioning the answer.\n\n"
        "##\nInstruction: Arrange the steps of CPR in the correct order.\n"
        "Need retrieval?: [Yes]\n"
        "Explanation: This task benefits from consulting reliable medical sources to ensure the steps are accurate and up to date.\n\n"
        "##\nInstruction: Explain the process of insulin regulation in the human body.\n"
        "Need retrieval?: [Yes]\n"
        "Explanation: This instruction asks for a detailed description of a scientific concept, and it is highly likely that we can find a reliable and useful document to support the response.\n\n"
        "##\nInstruction:{instruction}\n"
        "Need retrieval?: "
    ),

    "multi_retrieval": (
        "instruction:You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. If the output sentence can be verified solely with the evidence or doesn’t require any verification, respond with [No Retrieval]. If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments.\n\n"
        "input:Explain the role of insulin in glucose metabolism.\n"
        "Preceding sentences: Insulin is a hormone produced by the pancreas that plays a crucial role in regulating blood glucose levels. It allows cells to take in glucose to be used for energy or stored for future use.\n"
        "evidence: Insulin\nInsulin is a peptide hormone produced by beta cells of the pancreatic islets; it is considered to be the main anabolic hormone of the body. It regulates the metabolism of carbohydrates, fats, and protein by promoting the absorption of glucose from the blood into liver, fat, and skeletal muscle cells.\n"
        "output: [Retrieval]\n"
        "input: {input}\n"
        "Preceding sentences: {Output}\n"
        "evidence: {evidence}\n"
        "output:\n"),

    "multi_retrieval_no_preceding": (
        "You will be provided with an instruction, evidence, output, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. If the output sentence can be verified solely with the evidence or does not require any verification, respond with [No Retrieval]. If additional information is needed to verify the output sentence, respond with [Retrieval].\n\n"
        "# Example:\n\n"
        "Input:Explain the role of insulin in glucose metabolism.\n"
        "Preceding sentences: Insulin is a hormone produced by the pancreas that plays a crucial role in regulating blood glucose levels. It allows cells to take in glucose to be used for energy or stored for future use.\n"
        "Evidence: Insulin is a peptide hormone produced by beta cells of the pancreatic islets; it is considered to be the main anabolic hormone of the body. It regulates the metabolism of carbohydrates, fats, and protein by promoting the absorption of glucose from the blood into liver, fat, and skeletal muscle cells.\n"
        "Response is:\n"                
        "output: [Retrieval]\n"        
        "User Input:"
        "input: {input}\n"
        "Output:{Output}"
        "evidence: {evidence}\n"
        "Response:\n"
        
        
    ),
    "multi_retrieval_three_way": (
        "instruction:instruction:You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. There are three cases:\n"
        "- If the output sentence can be verified solely with the evidence, then respond with [Continue to Use Evidence]. \n"
        "- If the sentence doesn't require any factual verification (e.g., a subjective sentence or a sentence about common sense), then respond with [No Retrieval]. \n"
        "- If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments. \n\n"
        "input:Explain the role of insulin in glucose metabolism.\n"
        "Preceding sentences: Insulin is a hormone produced by the pancreas that plays a crucial role in regulating blood glucose levels. It allows cells to take in glucose to be used for energy or stored for future use.\n"
        "evidence: Insulin\nInsulin is a peptide hormone produced by beta cells of the pancreatic islets; it is considered to be the main anabolic hormone of the body. It regulates the metabolism of carbohydrates, fats, and protein by promoting the absorption of glucose from the blood into liver, fat, and skeletal muscle cells.\n"
        "output: [Retrieval]\n"
        "##\n input: {input}\n"
        "Preceding sentences: {Output}\n"
        "evidence: {evidence}\n"
        "output:\n"
    ),
    "multi_retrieval_three_way_no_preceding": (
        "instruction:You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. There are three cases:\n"
        "- If the output sentence can be verified solely with the evidence, then respond with [Continue to Use Evidence]. \n"
        "- If the sentence doesn't require any factual verification (e.g., a subjective sentence or a sentence about common sense), then respond with [No Retrieval]. \n"
        "- If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments. \n\n"
        "input: Explain the role of insulin in glucose metabolism.\n"
        "Preceding sentences: Insulin is a hormone produced by the pancreas that plays a crucial role in regulating blood glucose levels. It allows cells to take in glucose to be used for energy or stored for future use.\n"
        "evidence: Insulin\nInsulin is a peptide hormone produced by beta cells of the pancreatic islets; it is considered to be the main anabolic hormone of the body. It regulates the metabolism of carbohydrates, fats, and protein by promoting the absorption of glucose from the blood into liver, fat, and skeletal muscle cells.\n"
        "output: [Retrieval]\n"
        "input: {input}\n Preceding sentences:{Output}"
        "evidence: {evidence}\n"
        "output: \n"),

      "retrieval":   (
    "instruction : When provided with instruction, please evaluate whether seeking additional information from external sources such as the web (e.g., Wikipedia) aids in producing a more comprehensive response. Respond with either [Retrieval] or [No Retrieval]."
    "#Example:"
    "Input: Explain the role of insulin in glucose metabolism."
    "Response: [Retrieval]"
    "Task: retrieval"
    "Input: {input}\n"
    "Task: retrieval"
    "Response:\n"
        )



}


def create_result(output):
    return {"output": output}


OUTPUT_INSTRUCTIONS = "When provided with instruction, please evaluate whether seeking additional information from external sources such as the web (e.g., Wikipedia) aids in producing a more comprehensive response. Respond with either [Retrieval] or [No Retrieval]."


OUTPUT_INPUT = (
    "instruction: {instruction}\n"
    "input:{input}"
    "evidence: {evidence}\n"
    "output:{output}"
)

import ast
class Retrieval(Prompt):

    def format(self, data: dict[str, str]) -> str:
        return PROMPT_DICT["retrieval"].format_map(data)

    def parse(self, result: str, row: Any) -> str:
        
        data = ast.literal_eval(row["evidence"])
        
        if 'contexts' in data:
            row["evidence"]= data['contexts']

        return {"token": clean_text(result) , 'instruction':retrieval,"task" :'retrieval',
                "input": row["input"]}
