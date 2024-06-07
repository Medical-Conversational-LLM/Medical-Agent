import numpy as np
import string
import re
from collections import Counter
import re 
from sentence_transformers import SentenceTransformer, util
def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def accuracy(preds, labels):
    print('preds, labels ===========', preds, labels)
    match_count = 0
    for pred, label in zip(preds, labels):
        target = label[0]
        print("LABEL:{}".format(label))
        print("PRED:{}".format(pred))
        print("TARGET:{}".format(target))
        
        if pred == target:
            match_count += 1
            print('match_count' , match_count)

    return 100 * (match_count / len(preds))


def f1(decoded_preds, decoded_labels):
    f1_all = []
    for prediction, answers in zip(decoded_preds, decoded_labels):
        if type(answers) == list:
            if len(answers) == 0:
                return 0
            f1_all.append(np.max([qa_f1_score(prediction, gt)
                          for gt in answers]))
        else:
            f1_all.append(qa_f1_score(prediction, answers))
    return 100 * np.mean(f1_all)


def qa_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def find_entity_tags(sentence):
    entity_regex = r'(.+?)(?=\s<|$)'
    tag_regex = r'<(.+?)>'
    entity_names = re.findall(entity_regex, sentence)
    tags = re.findall(tag_regex, sentence)

    results = {}
    for entity, tag in zip(entity_names, tags):
        if "<" in entity:
            results[entity.split("> ")[1]] = tag
        else:
            results[entity] = tag
    return results


def match(prediction,ground_truth,threshold=0.5):
    print('prediction = ',prediction,'ground_truth = ',ground_truth)
    # Load pre-trained BERT model
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    # Encode sentences into fixed-size vectors
    embeddings1 = model.encode(prediction, convert_to_tensor=True)
    embeddings2 = model.encode(ground_truth, convert_to_tensor=True)

    # Compute cosine similarity between the sentence embeddings
    cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2).item()
    print("Cosine Similarity:", cosine_score)
    if cosine_score >= threshold:
        return 1
    return cosine_score




# def match(prediction, ground_truth):
#     for gt in ground_truth:
#         print('gt' , gt)
#         if gt in prediction:
#             return 1
#     return 0

# def match(prediction,ground_truth,threshold=0.8):

    # print('prediction = ',prediction,'ground_truth = ',ground_truth)
    # # Ensure ground_truth is a list
    # if isinstance(ground_truth, float):
    #     ground_truth = [ground_truth]
    # elif isinstance(ground_truth, str):
    #     ground_truth = [ground_truth]
    # elif not isinstance(ground_truth, list):
    #     raise TypeError(f"Expected ground_truth to be a list, float, or str, but got {type(ground_truth)}")
    # # Load a pre-trained SpaCy model
    # nlp = spacy.load("en_core_web_md")
    # # Process the prediction with the spaCy model
    # pred_doc = nlp(prediction)
    
    # for gt in ground_truth:
    #     # Process each ground truth with the spaCy model
    #     gt_doc = nlp(gt)
    #     similarity = pred_doc.similarity(gt_doc)
    #     print('similarity' ,similarity)
    #     if similarity >= threshold:
    #         return 1
    # return 0



