from datasets import load_dataset
import pandas as pd


dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")


df = pd.DataFrame(dataset)

df.to_csv("./storage/datasets/pubmed2.csv", index=False)
