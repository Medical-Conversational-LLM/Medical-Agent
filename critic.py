import pandas as pd
import ast

# Assuming the data is in a CSV file
# Adjust the 'filepath' variable to the path of your dataset
filepath = './storage/datasets/updated_PubMedQA_pqa_artificial.csv'

# Load the dataset
df = pd.read_csv(filepath)

# Define a function to extract the 'contexts' from the JSON string
def extract_contexts(json_str):
    # Use ast.literal_eval to safely evaluate the string as a dictionary
    data = ast.literal_eval(json_str)
    if 'contexts' in data:
        return data['contexts']
    else:
        return []

# Apply the function to the 'contexts' column (which is the third column, index 2)
df['contexts'] = df.iloc[:, 2].apply(extract_contexts)

# Now the 'contexts' column will have the extracted contexts
(df['contexts'])

# Save the updated dataframe to a new CSV file
df.to_csv('./storage/playground/updated_dataset.csv', index=False)
