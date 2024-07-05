# Medical Conversational Evidence Based Self-reflective Engine
Large language models (LLMs) have recently shown capabilities in answering conversational questions and generating human-like text, which led to interest in healthcare applications. Although not designed specifically for clinical use, it have the potential to transform healthcare delivery by improving documentation of patient reports, enhancing diagnostic accuracy, and supporting various clinical tasks.

# Demo:
http://134.91.35.190:8040/

# Content
#### [Setup](#setup)
#### [LLM](https://github.com/HlaHusain/Medical-Conversational-LLM/tree/main/lm#readme)
#### [Frontend](https://github.com/HlaHusain/Medical-Conversational-LLM/tree/main/client#readme)

# Setup & Installation

```
conda env create -f environment.yml
conda activate self-reflective-llm
```

### Create embeddings
```
python create_medical_records_index.py
```

### Web Inference
```
./server.sh
```
### using gunicorn
```
./server.sh prod
```

### Test inference
```
python run.py
```

### Web client
requires node 16+
```
cd client
npm install
npm run dev --  --host
```
