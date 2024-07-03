import evaluate
  

import json

# Load data from JSON file
with open('output1.json', 'r') as file:
    data = json.load(file)

# Access specific results (assuming 'predictions' and 'references' are keys in JSON)
predictions = data['predictions']
references = data['references']

# Print or use the results
for pred, ref in zip(predictions, references):
    print(f"Prediction: {pred}")
    print(f"Reference: {ref}")
    print()

rouge = evaluate.load('rouge')
rouge = rouge.compute(predictions=predictions,references=references)
print('rouge =====' , rouge)  
