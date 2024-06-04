import json
import random
from reader.line_to_json import LineToJson
# Load the JSON data from a file
reader = LineToJson(
    'storage/output/test.json'
)

data = [item for item in reader.read()]


# random.shuffle(data)
 

with open('storage/output/randomized_dataset3.json', 'w') as file:
    json.dump(data, file, indent=4)