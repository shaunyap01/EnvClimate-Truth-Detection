import pandas as pd
from openai import OpenAI

# Set the API key and model name
MODEL = "gpt-4o"
client = OpenAI(api_key="YOUR_API_KEY_HERE")

# Define the number of times each statement will be evaluated
num_repeats = 10

# Initialize empty columns for storing each round of responses
for i in range(1, num_repeats + 1):
    df[f'gpt-4o_{i}'] = ''

# Iterate over each row in the dataset
for index, row in df.iterrows():
    statement = row['statement_clean']
    
    # Evaluate the statement multiple times
    for i in range(1, num_repeats + 1):
        try:
            # Query the model with a prompt asking to determine if the statement is True or False
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Please determine whether the following statement is 'True' or 'False'. Respond with only 'True' or 'False' and nothing else."},
                    {"role": "user", "content": f"Statement: {statement}"}
                ]
            )
            
            # Extract the model's response and ensure it's either 'True' or 'False'
            response = completion.choices[0].message.content.strip()
            if response.lower() == "true":
                df.at[index, f'gpt-4o_{i}'] = "True"
            elif response.lower() == "false":
                df.at[index, f'gpt-4o_{i}'] = "False"
            else:
                df.at[index, f'gpt-4o_{i}'] = "Uncertain"  # Handle unexpected output

        except Exception as e:
            df.at[index, f'gpt-4o_{i}'] = f"Error: {str(e)}"  # Handle any API or network errors

# Save the updated dataframe to a new CSV file
df.to_csv(gpt4o_truthfulness_evaluation.csv, index=False)
