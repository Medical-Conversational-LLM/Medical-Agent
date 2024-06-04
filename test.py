from data_collection.collector import Collector
from inference.chatgpt import Chatgpt
from inference.mocked_inference import MockedInference
from reader.csv import CSV
from prompt.utility import Utility
from prompt.relevance import Relevance
from prompt.retrieval import Retrieval
from prompt.supported import Supported
from prompt.generator import Genertator
from writer.json import JsonWriter
from inference.pool import Pool
from inference.groq import Groq
from cache.tiny_db import TinyDBCache

collector = Collector(
    inference=[
        # Groq(
        #     api_key="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",
        #     model="mixtral-8x7b-32768",
        # ),
        # Groq(
        #     api_key="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",
        #     model="gemma-7b-it",
        # ),
        Groq(
            api_key="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",
            model="llama3-70b-8192",
        ),
        # Chatgpt(api_key="sk-proj-QB9Fv5VB0UxmFbHFWqXhT3BlbkFJjr0YJDPlBU6TpuMvIpAE"),
        # MockedInference()
    ],
    prompt=[
        # Utility(),
        # Relevance(),
        # Retrieval(),
        # Supported(),
        Genertator(),
    ]
)

collector.collect(
    reader=CSV(
        "./storage/datasets/updated_PubMedQA_pqa_artificial.csv",
        skip_header=True,
        mapper=lambda row: dict(
            {
                "id": row[0],
                # "input": row[1],
                # "evidence": row[2],
                # "Output": row[3],
                # "target_output": row[3],
                "question": row[1],
                  "long_answer": row[3],
                    "context": row[2],
                      "final_decision": row[4],
                        "loe": row[5],
            }
        ),
    ),
    writer=JsonWriter("./storage/output/generator1.json"),
    slice_start=44000,
    slice=20000,
    random_prompt=True
)
