from datasets import load_dataset
import pandas as pd


retrieval_dataset = load_dataset("ccdv/pubmed-summarization")
retrieval_df = pd.DataFrame(retrieval_dataset["train"])
