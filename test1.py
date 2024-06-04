import json

# Define the path to your JSON file
input_filepath = './storage/output/critic.json'
output_filepath = './storage/playground/critic1.json'

# Load the data from the JSON file
with open(input_filepath, 'r') as infile:
    data = [json.loads(line) for line in infile]

# Filter out entries with the task "groudness"
filtered_data = [entry for entry in data if entry.get('task') != 'groudness']

# Save the filtered data back to a JSON file
with open(output_filepath, 'w') as outfile:
    for entry in filtered_data:
        json.dump(entry, outfile)
        outfile.write('\n')

print(f"Filtered data saved to {output_filepath}")
