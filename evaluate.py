from evaluation.evaluate import evaluate
import pandas as pd
if __name__ == "__main__":

    # model_id = "nvidia/Llama3-ChatQA-1.5-8B"
    model_id = 'HlaH/Llama3-ChatQA-Generator-PubMedQA'

    # df = pd.read_csv('qiaojin/PubMedQA_test_clean_fixed.csv')
    df = pd.read_csv('storage/datasets/PubMedQA_test_clean_fixed.csv')
    # Convert the DataFrame to a Hugging Face Dataset
    # df = Dataset.from_pandas(df)
    df = df[0:10]
    

    evaluate(df)
    
