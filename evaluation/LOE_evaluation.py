
from inference.pretrained import Pretrained
import random
import torch
import numpy as np
from tqdm import tqdm
import ast
import re
import pandas as pd
from .evaluate_self_reflection import fix_context, get_model, get_formatted_input,filter_by_high_loe
import evaluate

from .metrics import match,similarity
from .utils import control_tokens

# GENERATOR_MODEL_ID = 'Llama-2-7b-ChatQA-Generator-PubMedQA'
GENERATOR_MODEL_ID = 'HlaH/Llama3-ChatQA-Generator-PubMedQA'

def loe_evaluation(df):  
    torch.cuda.empty_cache()
        # Filter the dataset based on the high LoE level
    df = filter_by_high_loe(df, 3)
    results=[]
    references=[]
    correct = 0
    sim=0
    for index, row in tqdm(df.iterrows(), total=len(df)):
        references.append(row["answer"])
        prompt = row["question"]
        evidences = fix_context(row["context"])
        loe = row["loe"]

        relevant_chunks = str(evidences)
        stuffed_context_with_loe = "".join(
            f"Level of Evidence {int(loe) if pd.notna(loe) else 'Unknown'} , {relevant_chunks}"
        )
        evidence_augmented_inputs = get_formatted_input(
            prompt, stuffed_context_with_loe)

        model = get_model(GENERATOR_MODEL_ID)
        
        answer, preds = model.completion(
            evidence_augmented_inputs, output_scores=True, max_new_tokens=128)
        
        pred_log_probs = preds.scores[0]
        results.append(answer)
        print('pred_text',answer)


        # if match(answer, row["answer"]) == 1:
        #     correct += 1

        if similarity(answer, row["answer"]) == 1:
            sim += 1    
    

    rouge = evaluate.load('rouge')
    rouge = rouge.compute(predictions=results,references=references)


    total = len(df)
    # acc = correct / total
    similarity_acc = sim / total

    print(f"Total: {total}, Correct: {correct} , similarity_acc:{similarity_acc}")
    

def trim_all_responses(responses):
    trimmed_responses = []
    for response in responses:
        # Assuming each response ends properly with a punctuation mark
        trimmed = trim_to_last_full_sentence(response)
        trimmed_responses.append(trimmed)
    return trimmed_responses


def trim_to_last_full_sentence(text):
    # Finds all occurrences of sentence-ending punctuation followed by a space or the end of the text
    sentences = re.findall(r'.*?[.!?](?:\s|$)', text)
    return ''.join(sentences)    