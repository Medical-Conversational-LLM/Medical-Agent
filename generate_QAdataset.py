import pandas as pd
import openai

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Set up the OpenAI API key
openai.api_key = 'your_openai_api_key'
api_key="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",
model="llama3-70b-8192",
def transform_question(question, context, long_answer, final_decision, loe):
    # Prepare the prompt for GPT-4
    prompt = f"""
    I have a yes/no question that needs to be converted into an open-ended question. Here is the information:
    
    Question: {question}
    Context: {context}
    Long Answer: {long_answer}
    Final Decision: {final_decision}
    Level of Evidence (LOE): {loe}
    
    Convert the above yes/no question into an open-ended question and use the provided information to generate a comprehensive answer.
    """
    
    # Call the OpenAI API
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7
    )
    
    # Extract the generated open-ended question and answer
    generated_text = response.choices[0].text.strip()
    
    return generated_text

# Create a new dataframe to store the transformed data
transformed_data = []

# Process each row in the dataset
for index, row in df.iterrows():
    transformed_entry = transform_question(
        row['question'], 
        row['context'], 
        row['long_answer'], 
        row['final_decision'], 
        row['loe']
    )
    
    transformed_data.append({
        'pubid': row['pubid'],
        'original_question': row['question'],
        'open_ended_question_and_answer': transformed_entry
    })

# Create a new DataFrame with the transformed data
transformed_df = pd.DataFrame(transformed_data)

# Save the transformed data to a new CSV file
transformed_df.to_csv('transformed_dataset.csv', index=False)

print("Transformation complete and saved to 'transformed_dataset.csv'.")
