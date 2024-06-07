
# import spacy
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import random
import torch
from peft import PeftModel, PeftConfig
import os
import numpy as np
import openai
from tqdm import tqdm
import json
import argparse
import ast
import re
from collections import Counter
import string
import sys
import time
from metrics import match, accuracy
from utils import PROMPT_DICT, TASK_INST, load_jsonlines, control_tokens, load_special_tokens
seed = 633

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
base_model = AutoModelForCausalLM.from_pretrained(
                "nvidia/Llama3-ChatQA-1.5-8B")


class SamplingParams:
    def __init__(self, temperature=1.0, top_p=1.0, max_tokens=50, logprobs=None):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.logprobs = logprobs

    def __repr__(self):
        return f"SamplingParams(temperature={self.temperature}, top_p={self.top_p}, max_tokens={self.max_tokens}, logprobs={self.logprobs})"


model_cache = {}
def get_model(model_id):

    if model_id in model_cache:
        return model_cache[model_id]
    
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("LOADING MODEL {}".format(model_id))
    model = PeftModel.from_pretrained(base_model,
                                      model_id,
                                      torch_dtype=torch.bfloat16,
                                      attn_implementation="flash_attention_2"
                                      ).to("cuda")
    model = torch.compile(model)
    model_cache[model_id] = model

    print("LOADED MODEL {}".format(model_id))
    return model


torch.cuda.empty_cache()
# Choose the specific GPU (e.g., 'cuda:0' for the first GPU)
gpu_index = 1  # Change this to the index of the GPU you want to use
device = torch.device("cuda:0")


def postprocess_answer_option_conditioned(answer):
    for token in control_tokens:
        answer = answer.replace(token, "")

    if "</s>" in answer:
        answer = answer.replace("</s>", "")
    if "\n" in answer:
        answer = answer.replace("\n", "")

    if "" in answer:
        answer = answer.replace("", "")

    return answer


def get_formatted_input(messages, context):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    # instruction = "Please give a full and complete answer for the question."
    if context == 'No Retrieval':
        print('context = ', context)
        instruction = f"""Your task is to evaluate whether the input question requires seeking additional information from other sources to produce a more comprehensive response.If additional information is needed to answer the input question, respond with [Retrieval]. If no additional information is needed, respond with [No Retrieval]"""
        conversation = '\n\n'.join(["User: " + messages]) + "\n\nAssistant:"
        formatted_input = system + "\n\n" + instruction + "\n\n" + conversation

    elif context != None:
        print('context = ', context)
        instruction = f""" Using the detailed information provided in the context below, answer the question and generate a response that strictly adheres to this information. Ensure that your answer is deeply grounded in the specifics of the context, does not include extraneous details not supported by the context ."""
        conversation = '\n\n'.join(["User: " + messages]) + "\n\nAssistant:"
        formatted_input = system + "\n\n" + instruction + \
            "\n\n" + "[Retrieval]"+context + conversation

    else:
        print('context = ', context)
        instruction = f"""Your task is answer the question and generate a response that strictly adheres to the question without any external resources or context."""
        conversation = '\n\n'.join(["User: " + messages]) + "\n\nAssistant:"
        formatted_input = system + "\n\n" + instruction + "\n\n" + conversation
    # for item in messages:
    #     if item['role'] == "user":
    #         ## only apply this instruction for the first user turn
    #         item['content'] = instruction + " " + item['content']
    #         break

    return formatted_input


def call_model_rerank_w_scores_batch(prompt, evidences,  max_new_tokens=15,
                                     ret_tokens=None, rel_tokens=None, grd_tokens=None, ut_tokens=None,
                                     use_seqscore=False, threshold=0.5,
                                     w_rel=1.0, w_sup=1.0, w_use=0.5, mode="adaptive_retrieval", closed=False):
    results = {}
    if mode != "always_retrieve":
        print('--not always_retrieve')
        
        # evidences = "\n\n".join(evidences)
        model_id = 'HlaH/Llama3-ChatQA-Retriever-PubMedQA'
        formatted_input = get_formatted_input(prompt, 'No Retrieval')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = get_model(model_id)

        # print('formatted_input' , formatted_input)
        # Move the tokenizer to the same device as the model

        # Assuming formatted_input is already defined
        tokenized_prompt = tokenizer(
            tokenizer.bos_token + formatted_input, return_tensors="pt").to(model.device)

         
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        output = model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=128, eos_token_id=terminators, return_dict_in_generate=True,
                                output_scores=True)
        sequences = output.sequences
        response = sequences[0][tokenized_prompt.input_ids.shape[-1]:]

        # Decode the generated tokens to text
        pred_text = tokenizer.decode(response, skip_special_tokens=True)
        print('pred_text ======', pred_text)
        pred_log_probs = output.scores[0]
        print('pred_log_probs ======= ', pred_log_probs)
        results["no_retrieval"] = pred_text

    # save relevance token scores
    if mode == "always_retrieve":
        print('--always_retrieve')
        do_retrieve = True

    elif mode == "no_retrieval":
        print('--no_retrieval')
        do_retrieve = False

    if pred_text.strip() == '[Retrieval]':
        print('--use Retriver ')
        do_retrieve = True
    elif pred_text.strip() == '[No Retrieval]':
        print('--use Retriver ')
        do_retrieve = False
    else:
        print('--use threshold ')
        if threshold is not None:
            score_dict = {}
            for tok, id in ret_tokens.items():
                if id not in pred_log_probs[0]:
                    score_dict[tok] = -100
                prob = pred_log_probs[0][id]
                score_dict[tok] = float(prob.mean().item())
            print('Retrieval =',
                  score_dict["[Retrieval]"], score_dict["[No Retrieval]"])
            do_retrieve = score_dict["[Retrieval]"] / (
                score_dict["[Retrieval]"] + score_dict["[No Retrieval]"]) > threshold
        else:
            do_retrieve = "[Retrieval]" in pred_text

        print(threshold, 'do_retrieve========', do_retrieve, 'score_dict', score_dict["[Retrieval]"] / (
            score_dict["[Retrieval]"] + score_dict["[No Retrieval]"]))

    if do_retrieve is True:
        evidence_augmented_inputs = get_formatted_input(prompt, evidences)
        # evidence_augmented_inputs =formatted_input + "[Retrieval]"+evidences
        # print('evidence_augmented_inputs' , evidence_augmented_inputs)
        
        # preds = model.generate(evidence_augmented_inputs, sampling_params)

        model_id = 'HlaH/Llama3-ChatQA-Generator-PubMedQA'
    
        model = get_model(model_id)

        tokenized_prompt = tokenizer(
            tokenizer.bos_token + evidence_augmented_inputs, return_tensors="pt").to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        preds = model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=128, eos_token_id=terminators, return_dict_in_generate=True,
                               output_scores=True)
        sequences = preds.sequences
        response = sequences[0][tokenized_prompt.input_ids.shape[-1]:]
        pred = tokenizer.decode(response, skip_special_tokens=True)
        relevance_score_dict = {}
        grd_score_dict = {}
        ut_score_dict = {}
        overall_scores = {}

# Iterate over the generated sequences
        for p_idx, (sequence, scores) in enumerate(zip(preds.sequences, preds.scores)):
            pred_token_ids = sequence.tolist()
            pred_text = tokenizer.decode(sequence, skip_special_tokens=True)

            # Convert scores (logits) to log probabilities
            pred_log_probs = [score.log_softmax(dim=-1) for score in scores]

            # Calculate sequence score
            seq_score = sum([log_prob[token_id].item() if token_id < log_prob.size(-1) else -100 for token_id,
                            log_prob in zip(pred_token_ids, pred_log_probs)]) / max(len(pred_token_ids), 1)

            relevance_score_dict.setdefault(p_idx, {})
            grd_score_dict.setdefault(p_idx, {})
            ut_score_dict.setdefault(p_idx, {})

            # Compute reward scores
            for tok, id in rel_tokens.items():
                prob = pred_log_probs[0][id].item(
                ) if id < pred_log_probs[0].size(-1) else -100
                relevance_score_dict[p_idx][tok] = np.exp(float(prob))

            if grd_tokens is not None:
                groundness_token_appear_indices = [tok_idx for tok_idx, tok in enumerate(
                    pred_token_ids) if tok in grd_tokens.values()]
                if groundness_token_appear_indices:
                    idx = groundness_token_appear_indices[0]
                    for token, token_id in grd_tokens.items():
                        if idx < len(pred_log_probs):
                            prob = pred_log_probs[idx][token_id].item(
                            ) if token_id < pred_log_probs[idx].size(-1) else -100
                            grd_score_dict[p_idx][token] = np.exp(float(prob))

            if ut_tokens is not None:
                utility_token_appear_indices = [tok_idx for tok_idx, tok in enumerate(
                    pred_token_ids) if tok in ut_tokens.values()]
                if utility_token_appear_indices:
                    idx = utility_token_appear_indices[0]
                    for token, token_id in ut_tokens.items():
                        if idx < len(pred_log_probs):
                            prob = pred_log_probs[idx][token_id].item(
                            ) if token_id < pred_log_probs[idx].size(-1) else -100
                            ut_score_dict[p_idx][token] = np.exp(float(prob))

            relevance_score = relevance_score_dict[p_idx].get(
                "[Relevant]", 0) / max(np.sum(list(relevance_score_dict[p_idx].values())), 1)

            if len(grd_score_dict[p_idx]) == 3:
                gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
                ground_score = (grd_score_dict[p_idx].get("[Fully supported]", 0) / gt_sum) + 0.5 * (
                    grd_score_dict[p_idx].get("[Partially supported]", 0) / gt_sum)
            else:
                ground_score = 0.0

            if len(ut_score_dict[p_idx]) == 5:
                ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
                ut_scores = [-1, -0.5, 0, 0.5, 1]
                utility_score = np.sum([ut_scores[i] * (ut_score_dict[p_idx].get(
                    "[Utility:{}]".format(i + 1), 0) / ut_sum) for i in range(len(ut_scores))])
            else:
                utility_score = 0.0

            use_seqscore = True
            w_rel = 1.0
            w_sup = 1.0
            w_use = 1.0

            if use_seqscore:
                final_score = np.exp(seq_score) + w_rel * relevance_score + \
                    w_sup * ground_score + w_use * utility_score
            else:
                final_score = w_rel * relevance_score + \
                    w_sup * ground_score + w_use * utility_score

            overall_scores[p_idx] = {"final_score": final_score,
                                     "relevance_score": relevance_score,
                                     "ground_score": ground_score,
                                     "utility_score": utility_score,
                                     "relevance_score_dict": relevance_score_dict,
                                     "grd_score_dict": grd_score_dict,
                                     "ut_score_dict": utility_score}

        results = {"retrieval_{}".format(
            p_idx): {"pred": pred_text, "score": final_score} for p_idx in overall_scores}
        # print('results === ' , results)
        # Iterate over the generated sequences
        # for p_idx, (sequence, scores) in enumerate(zip(preds.sequences, preds.scores)):
        #     pred_token_ids = sequence.tolist()
        #     pred_text = tokenizer.decode(sequence, skip_special_tokens=True)
        #     pred_log_probs = scores  # this gives the log probabilities for each token

        #     # Calculate sequence score
        #     seq_score = sum(pred_log_probs) / max(len(pred_token_ids), 1)

        #     relevance_score_dict.setdefault(p_idx, {})
        #     grd_score_dict.setdefault(p_idx, {})
        #     ut_score_dict.setdefault(p_idx, {})
        #     # Compute reward scores
        #     for tok, id in rel_tokens.items():
        #         prob = pred_log_probs[0][id] if id in pred_log_probs[0] else -100
        #         relevance_score_dict[p_idx][tok] = np.exp(float(prob))

        #     if grd_tokens is not None:
        #         groundness_token_appear_indices = []
        #         for tok_idx, tok in enumerate(pred_token_ids):
        #             if tok in list(grd_tokens.values()):
        #                 groundness_token_appear_indices.append(tok_idx)
        #                 break
        #         if len(groundness_token_appear_indices) > 0:
        #             idx = groundness_token_appear_indices[0]
        #             for token, token_id in grd_tokens.items():
        #                 prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
        #                 grd_score_dict[p_idx][token] = np.exp(float(prob))

        #     if ut_tokens is not None:
        #         utility_token_appear_indices = []
        #         for tok_idx, tok in enumerate(pred_token_ids):
        #             if tok in list(ut_tokens.values()):
        #                 utility_token_appear_indices.append(tok_idx)
        #         if len(utility_token_appear_indices) > 0:
        #             idx = utility_token_appear_indices[0]
        #             for token, token_id in ut_tokens.items():
        #                 prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
        #                 ut_score_dict[p_idx][token] = np.exp(float(prob))

        #     relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (
        #         np.sum(list(relevance_score_dict[p_idx].values())))

        #     if len(grd_score_dict[p_idx]) == 3:
        #         gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
        #         ground_score = (grd_score_dict[p_idx]["[Fully supported]"] / gt_sum) + 0.5 * (
        #             grd_score_dict[p_idx]["[Partially supported]"] / gt_sum)
        #     else:
        #         ground_score = 0.0

        #     if len(ut_score_dict[p_idx]) == 5:
        #         ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
        #         ut_scores = [-1, -0.5, 0, 0.5, 1]
        #         utility_score = np.sum(
        #             [ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i+1)] / ut_sum) for i in range(len(ut_scores))])
        #     else:
        #         utility_score = 0.0

        #     if use_seqscore is True:
        #         final_score = np.exp(seq_score) + w_rel * relevance_score + \
        #             w_sup * ground_score + w_use * utility_score
        #     else:
        #         final_score = w_rel * relevance_score + \
        #             w_sup * ground_score + w_use * utility_score

        #     overall_scores[p_idx] = {"final_score": final_score,
        #                              "relevance_score": relevance_score,
        #                              "ground_score": ground_score,
        #                              "utility_score": utility_score,
        #                              "relevance_score_dict": relevance_score_dict,
        #                              "grd_score_dict": grd_score_dict,
        #                              "ut_score_dict": utility_score}
        #     results["retrieval_{}".format(p_idx)] = {
        #         "pred": pred_text, "score": final_score, "ctx": evidences[p_idx]}

    else:
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
        prompt += "[No Retrieval]"
        # print('No Retrieval prompt = ' , prompt)
        formatted_input = get_formatted_input(prompt, None)
        # print('formatted_input = ', formatted_input
        # )

        model_id = 'HlaH/Llama3-ChatQA-Generator-PubMedQA'
        model = get_model(model_id)

        # Move the tokenizer to the same device as the model

        # Assuming formatted_input is already defined
        tokenized_prompt = tokenizer(
            tokenizer.bos_token + formatted_input, return_tensors="pt").to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        output = model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=128, eos_token_id=terminators, return_dict_in_generate=True,
                                output_scores=True)

        # Extract generated sequences
        sequences = output.sequences
        response = sequences[0][tokenized_prompt.input_ids.shape[-1]:]
        pred = tokenizer.decode(response, skip_special_tokens=True)

    # Aggregating answers
    if len(results) == 1:
        # postprocessed_pred = postprocess_answer_option_conditioned(pred)
        return pred, results, do_retrieve
    else:
        print('answer2score')
        answer2score = {}
        if closed is True:

            for key, result in results.items():
                if key == "no_retrieval":
                    continue
                answer = postprocess_answer_option_conditioned(result["pred"])
                score = result["score"]
                answer2score.setdefault(answer, 0)
                answer2score[answer] += score
            sorted_answers = sorted(
                answer2score.items(), key=lambda x: x[1], reverse=True)
            best_option = sorted_answers[0][0]

        else:
            path2score = {key: item["score"] for key,
                          item in results.items() if key != "no_retrieval"}
            best_path = sorted(path2score.items(),
                               key=lambda x: x[1], reverse=True)[0][0]
            best_option = results[best_path]["pred"]
        return best_option, results, do_retrieve


def run_pipeline(data, ret_tokens, rel_tokens, grd_tokens=None, ut_tokens=None, max_new_tokens=1, use_seqscore=False, threshold=0.5,
                 w_rel=1.0, w_sup=1.0, w_use=0.5, mode="adaptive_retrieval", closed=False):

    correct = 0
    no_retrieval = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        # print('index' , index , '------------------------------------------')
        prompt = row["question"]
        # prompt ="""Please answer the following question: Read the bio below and try to give details on babe martin 's: - finalteam - finalyear - death place - statlabel - position - statvalue - throws - debutteam - bats - debutdate - death date - birth date - finaldate - name - birth place - debutyear Bio: `` babe '' martin -lrb- march 28 , 1920 -- august 1 , 2013 -rrb- was a major league baseball outfielder for the st. louis browns -lrb- 1944 -- 46 and 1953 -rrb- and a catcher for the boston red sox -lrb- 1948 -- 49 -rrb- . martin was born boris michael martinovich in seattle , washington to serbian immigrant parents . the martinovich family moved to zeigler , illinois when babe was year old and subsequently moved to st. louis , missouri after the death of babe 's father . he started his professional baseball career in 1940 and had a breakout year in 1944 with the toledo mud hens , batting .350 in 114 games . the following season , he joined the major league browns . he hit poorly and was sent back down to the minors . martin retired in 1954 . in 69 major league games , he had 2 home runs , 18 rbi , and a .214 batting average . in later years , martin still held a grudge against one-armed outfielder pete gray , who was a teammate in 1945 . `` the worst thing that happened to our ballclub in 1945 , which we should have won the pennant , was pete gray , '' he said . `` and honestly i think if we had n't had pete ... we could have won the pennant in 1945 . '' although martin 's batting average that season was actually 18 points lower than gray 's , martin was referring to pete gray 's fielding ability . because gray only had one arm , his throws back into the infield were slowed because he had to remove his glove from his one hand , get the ball , and throw into the infield . this slowed him down and allowed runners to advance more easily than they otherwise would have . the browns finished in third place in the american league , six games behind the detroit tigers . A:	"""
        evidences = fix_context(row["context"])

        # item["instruction"] = TASK_INST[task]
        # prompt, evidences = process_data_evidences(item, top_n=3)
        answer, res, retrieved = call_model_rerank_w_scores_batch(prompt, evidences,  max_new_tokens=max_new_tokens,
                                                                  ret_tokens=ret_tokens, rel_tokens=rel_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                                                  use_seqscore=use_seqscore, threshold=threshold,
                                                                  w_rel=w_rel, w_sup=w_sup, w_use=w_use, mode=mode, closed=closed)

        if retrieved:
            print("answer, retrieved", answer)
        else:
            no_retrieval += 1
            print("answer, no retrieval", answer)

        if match(answer, row["answer"]) == 1:
            correct += 1
        # print('match correct ==',correct,'answer ======' , answer, 'row["answer"] =====', row["answer"])
    total = len(data)
    acc = correct / total
    no_retrieval_pct = no_retrieval / total

    print(
        f"Total: {total}, Correct: {correct}, Accuracy: {acc:.4f}, No Retrievals: {no_retrieval_pct:.4f}")
    return acc


df = pd.read_csv('../storage/datasets/PubMedQA_test_clean_fixed.csv')
# Convert the DataFrame to a Hugging Face Dataset
# df = Dataset.from_pandas(df)


def fix_context(value):
    try:
        value = ast.literal_eval(value)

        return value[0]
    except:
        return value


df['context'] = df['context'].apply(fix_context)


def main(df):
    # preproc_data = preprocess_input_data(dataset, task=task)

    new_data = []
    # for index, row in tqdm(df.iterrows(), total=len(df)):
    #     print('teeeeeeeeeeeeeeest!!!!' , row['context'])

    #     context=fix_context(row['context'])

    #     new_data.append({"question": row['question']})
    #     new_data.append({"context": context[0]})
    #     new_data.append({"answer": row['answer']})

    # preproc_data = new_data
    # print('new_data len : ',new_data)

    ret_tokens = {"[Retrieval]": 1, "[No Retrieval]": 2}
    rel_tokens = {"[Relevant]": 3, "[Irrelevant]": 4}
    grd_tokens = {"[Fully supported]": 5,
                  "[Partially supported]": 6, "[Not supported]": 7}
    ut_tokens = {"[Utility:1]": 8, "[Utility:2]": 9,
                 "[Utility:3]": 10, "[Utility:4]": 11, "[Utility:5]": 12}

    accuracy = run_pipeline(df, ret_tokens, rel_tokens, grd_tokens, ut_tokens, max_new_tokens=1,
                            use_seqscore=False, threshold=0.5, w_rel=1.0, w_sup=1.0, w_use=0.5, mode="adaptive_retrieval", closed=True)
    # print(f"Task , completed with accuracy: {accuracy:.4f}")


if __name__ == "__main__":

    # model_id = "nvidia/Llama3-ChatQA-1.5-8B"
    model_id = 'HlaH/Llama3-ChatQA-Generator-PubMedQA'

    # df = pd.read_csv('qiaojin/PubMedQA_test_clean_fixed.csv')
    df = pd.read_csv('../storage/datasets/PubMedQA_test_clean_fixed.csv')
    # Convert the DataFrame to a Hugging Face Dataset
    # df = Dataset.from_pandas(df)
    df = df[0:10]

    main(df)
