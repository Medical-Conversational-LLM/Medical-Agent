GENERATOR_MODEL_ID = 'HlaH/Llama3-ChatQA-Generator-PubMedQA'
from tqdm import tqdm
from .metrics import match,f1
import pandas as pd
import re
import sacrebleu
from .evaluate_self_reflection import fix_context, get_model, get_formatted_input
from transformers import AutoTokenizer

def evaluate_generator(df):

    df=df[0:300]
    references=[]
    predictions=[]
    correct = 0
    for index, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row["question"]
        evidences = fix_context(row["context"])
        loe=row["loe"]

        relevant_chunks= str(evidences)
        stuffed_context_with_loe = "".join(
            f"Level of Evidence {int(loe) if pd.notna(loe) else 'Unknown'} , {relevant_chunks}"
        )
        evidence_augmented_inputs = get_formatted_input(prompt, stuffed_context_with_loe)

        model = get_model(GENERATOR_MODEL_ID)
        print('evidence_augmented_inputs == ',evidence_augmented_inputs)

        pred_text, preds = model.completion(
            evidence_augmented_inputs, output_scores=True,max_new_tokens=512)
        
        predictions.append(pred_text)
        references.append(evidences) 

        if match(pred_text, row["answer"]) == 1:
            correct += 1


        # print(preds)   

        # model_id = "HlaH/Llama3-ChatQA-Generator-PubMedQA"  # Replace with your model ID
        # tokenizer = AutoTokenizer.from_pretrained(model_id) 
        # if tokenizer.pad_token is None:
        #    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
       
        # encoded_input = tokenizer(row["answer"], return_tensors="pt", truncation=True, padding=True)
        # encoded_preds = tokenizer(pred_text, return_tensors="pt", truncation=True, padding=True)
        # # print('encoded_input' , encoded_input)
        # f1_res = f1(encoded_preds, encoded_input)
        # print('f1_res ======= ' , f1_res)

        
    total = len(df)
    acc = correct / total

    print(
        f"Total: {total}, Correct: {correct}, Accuracy: {acc:.4f}")
    
    # cleaned_answers= clean_and_extract_answers(predictions)
    trimmed_text = trim_all_responses(predictions)
    # pred_tokens = normalize_text(trimmed_text).split()
    # truth_tokens = normalize_text(predictions).split()

    print('predictions =' ,  predictions, ' \n \n references = ', references)
    
    return predictions,  references  



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


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        text = " ".join(text)
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def blue_score(predictions, references):

    # Compute BLEU scores
    sacrebleu_results = sacrebleu.corpus_bleu(predictions, references)

    # Access BLEU score
    bleu_score = sacrebleu_results.score
    print( bleu_score )
    return bleu_score


