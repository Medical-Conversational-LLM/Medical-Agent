from datasets import load_dataset
import pandas as pd


 
dataset = load_dataset("ccdv/pubmed-summarization", trust_remote_code=True)



df = pd.DataFrame.from_dict(dataset, orient='index')
df = df.transpose()

df.to_csv("./storage/datasets/medical-records.csv", index=False)
