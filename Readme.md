
# Setup 

```
conda env create -f environment.yml
conda activate self-reflective-llm
```

# Create embeddings
```
python create_medical_records_index.py
```

# Web Inference
```
./server.sh
```


# Test inference
```
python run.py
```

# Web client
requires node 16+
```
cd client
npm install
npm run dev --  --host
```

# Workflow
![graph](assets/graph.svg)
