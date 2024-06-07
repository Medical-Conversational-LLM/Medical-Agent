from inference.pretrained import Pretrained
import random
import torch
import numpy as np
from tqdm import tqdm
import ast

from .metrics import match
from .utils import control_tokens

GENERATOR_MODEL_ID = 'HlaH/Llama3-ChatQA-Generator-PubMedQA'
RETRIEVER_MODEL_ID = 'HlaH/Llama3-ChatQA-Retriever-PubMedQA'

seed = 633
torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

 
model_cache = {}
def get_model(model_id):

    if model_id in model_cache:
        return model_cache[model_id]

    model = Pretrained(model_id)

    model_cache[model_id] = model

    return model

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
    prompt = [
        {
            "role": "system",
            "content": (
                "System: This is a chat between a user and an artificial intelligence assistant."
                "The assistant gives helpful, detailed, and polite answers to the user's questions."
            )
        }
    ]

    if context == 'No Retrieval':
        prompt.append({
            "role": "system",
            "content": (
                "Your task is to evaluate whether the input question requires"
                "seeking additional information from other sources to produce a "
                "more comprehensive response.If additional information is needed to answer the input question, "
                "respond with [Retrieval]. If no additional information is needed, respond with [No Retrieval]"
            )
        })
    elif context != None:
        prompt.append(
            {
                "role": "system",
                "content": (
                    "Using the detailed information provided in the context below, "
                    "answer the question and generate a response that strictly adheres to this information. "
                    "Ensure that your answer is deeply grounded in the specifics of the context, "
                    "does not include extraneous details not supported by the context ."
                    "\n\n"
                    "Context:\n{context}"
                ).format(context=context)
            }
        )
    else:
        prompt.append({
            "role": "user",
            "content": (
                "Your task is answer the question and generate a response that strictly "
                "adheres to the question without any external resources or context."
            )
        })

    prompt.append({
        "role": "user",
        "content": "{}\n Assistant:\n".format(messages)
    })

    return prompt


def call_model_rerank_w_scores_batch(prompt, evidences,  max_new_tokens=128,
                                     ret_tokens=None, rel_tokens=None, grd_tokens=None, ut_tokens=None,
                                     use_seqscore=False, threshold=0.5,
                                     w_rel=1.0, w_sup=1.0, w_use=0.5, mode="adaptive_retrieval", closed=False):
    results = {}
    if mode != "always_retrieve":
        print('--not always_retrieve')

        formatted_input = get_formatted_input(prompt, 'No Retrieval')

        model = get_model(RETRIEVER_MODEL_ID)

        pred_text, output = model.completion(
            formatted_input, output_scores=True,max_new_tokens=max_new_tokens)
 
 
        pred_log_probs = output.scores[0] 
        results["no_retrieval"] = pred_text
 
    if mode == "always_retrieve":
        print('--always_retrieve')
        do_retrieve = True

    elif mode == "no_retrieval":
        print('--no_retrieval')
        do_retrieve = False

    if pred_text.strip() == '[Retrieval]':
        print('--use Retriver !!!!  ',pred_text.strip())
        do_retrieve = True
    elif pred_text.strip() == '[No Retrieval]':
        print('--use No Retriver ')
        do_retrieve = False
    elif 'Retrieval' in pred_text.strip():
        score_dict = {}
        for tok, id in ret_tokens.items():
            if id not in pred_log_probs[0]:
                score_dict[tok] = -100
            prob = pred_log_probs[0][id]
            score_dict[tok] = float(prob.mean().item())

        do_retrieve = score_dict["[Retrieval]"] > score_dict["[No Retrieval]"]
    elif 'Retrieval' not in pred_text.strip():
        do_retrieve = True
            
    else:
        print('--use threshold ')
        if threshold is not None:
            score_dict = {}
            for tok, id in ret_tokens.items():
                if id not in pred_log_probs[0]:
                    score_dict[tok] = -100
                prob = pred_log_probs[0][id]
                score_dict[tok] = float(prob.mean().item())
          
            do_retrieve = score_dict["[Retrieval]"] / (
                score_dict["[Retrieval]"] + score_dict["[No Retrieval]"]) > threshold
        else:
            do_retrieve = "[Retrieval]" in pred_text

    if do_retrieve is True:
        evidence_augmented_inputs = get_formatted_input(prompt, evidences)

        model = get_model(GENERATOR_MODEL_ID)
        print('evidence_augmented_inputs',evidence_augmented_inputs)

        pred, preds = model.completion(
            evidence_augmented_inputs, output_scores=True,max_new_tokens=max_new_tokens)

        relevance_score_dict = {}
        grd_score_dict = {}
        ut_score_dict = {}
        overall_scores = {}

        # Iterate over the generated sequences
        for p_idx, (sequence, scores) in enumerate(zip(preds.sequences, preds.scores)):
            pred_token_ids = sequence.tolist()
            pred_text = model.tokenizer.decode(
                sequence, skip_special_tokens=True)

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
    else:

        prompt += "[No Retrieval]"
        formatted_input = get_formatted_input(prompt, None)

        model = get_model(GENERATOR_MODEL_ID)

        pred, output = model.completion(formatted_input, output_scores=True,max_new_tokens=max_new_tokens)
 

    if len(results) == 1:
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


def run_pipeline(df, ret_tokens, rel_tokens, grd_tokens=None, ut_tokens=None, max_new_tokens=128, use_seqscore=False, threshold=0.5,
                 w_rel=1.0, w_sup=1.0, w_use=0.5, mode="adaptive_retrieval", closed=False):

    correct = 0
    no_retrieval = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):

        prompt = row["question"]

        evidences = fix_context(row["context"])

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

    total = len(df)
    acc = correct / total
    no_retrieval_pct = no_retrieval / total

    print(
        f"Total: {total}, Correct: {correct}, Accuracy: {acc:.4f}, No Retrievals: {no_retrieval_pct:.4f}")
    return acc


def fix_context(value):
    try:
        value = ast.literal_eval(value)

        return "\n\n".join(value)
    
    except:
        return value


def evaluate(df):
    df['context'] = df['context'].apply(fix_context)
    print('context 000 ',df['context'][5])

    ret_tokens = {"[Retrieval]": 1, "[No Retrieval]": 2}
    rel_tokens = {"[Relevant]": 3, "[Irrelevant]": 4}
    grd_tokens = {"[Fully supported]": 5,
                  "[Partially supported]": 6, "[Not supported]": 7}
    ut_tokens = {"[Utility:1]": 8, "[Utility:2]": 9,
                 "[Utility:3]": 10, "[Utility:4]": 11, "[Utility:5]": 12}

    accuracy = run_pipeline(df, ret_tokens, rel_tokens, grd_tokens, ut_tokens, max_new_tokens=128,
                            use_seqscore=False, threshold=0.5, w_rel=1.0, w_sup=1.0, w_use=0.5, mode="adaptive_retrieval", closed=True)
    print(f"Task , completed with accuracy: {accuracy:.4f}")
