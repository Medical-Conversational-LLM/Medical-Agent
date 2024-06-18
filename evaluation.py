# from evaluation.evaluate import evaluate
from evaluation.evaluate_generator import evaluate_generator
import pandas as pd
import torch
import os
from evaluation.LOE_evaluation import loe_evaluation
from evaluation.evaluate_self_reflection import evaluate_self_reflection
# import evaluate as evaluate_rouge
if __name__ == "__main__":
    # model_id = "nvidia/Llama3-ChatQA-1.5-8B"
    # model_id = 'HlaH/Llama3-ChatQA-Generator-PubMedQA'
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # df = pd.read_csv('qiaojin/PubMedQA_test_clean_fixed.csv')
    df = pd.read_csv('storage/datasets/PubMedQA_test_clean_fixed.csv')
    # Convert the DataFrame to a Hugging Face Dataset
    # df = Dataset.from_pandas(df)
    # df=df[0:]
    
    # evaluate_self_reflection(df)
    loe_evaluation(df)
    # pred,ref = evaluate_generator(df)
    
