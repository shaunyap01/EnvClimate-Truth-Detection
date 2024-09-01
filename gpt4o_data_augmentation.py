import pandas as pd
from openai import OpenAI

# Set the API key and model name
MODEL = "gpt-4o"
client = OpenAI(api_key="your_api_key_here")

# Define the number of times each statement will be evaluated
num_repeats = 3

# Initialize empty columns in the DataFrame for storing responses
for i in range(1, num_repeats + 1):
    df[f'gpt-4o_readability{i}'] = 0
    df[f'gpt-4o_subjectivity{i}'] = 0
    df[f'gpt-4o_polarity{i}'] = 0

# Iterate over each row in the dataset
for index, row in df.iterrows():
    statement = row['statement_clean']
    
    # Evaluate readability, subjectivity, and polarity multiple times
    for i in range(1, num_repeats + 1):
        
        # Readability score
        readability_response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Please provide only a single number between 0 and 100 to indicate the readability score of the following statement, where 0 is Very difficult (college graduate level) and 100 is Very easy (5th grade level). Do not include any additional text or commentary."},
                {"role": "user", "content": f"Statement: {statement}"}
            ]
        )
        
        readability_score = readability_response.choices[0].message.content.strip()
        try:
            score = int(readability_score)
            if 0 <= score <= 100:
                df.at[index, f'gpt-4o_readability{i}'] = score
            else:
                df.at[index, f'gpt-4o_readability{i}'] = None  # Handle out-of-range output
        except ValueError:
            df.at[index, f'gpt-4o_readability{i}'] = None  # Handle non-integer output

        # Subjectivity score
        subjectivity_response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Please provide a single subjectivity score as a number between 0 and 100 for the following statement, where 0 indicates that the text is very objective (factual) and 100 indicates that the text is very subjective (opinionated). Do not include any additional text or commentary."},
                {"role": "user", "content": f"Statement: {statement}"}
            ]
        )
        
        subjectivity_score = subjectivity_response.choices[0].message.content.strip()
        try:
            score = int(subjectivity_score)
            if 0 <= score <= 100:
                df.at[index, f'gpt-4o_subjectivity{i}'] = score
            else:
                df.at[index, f'gpt-4o_subjectivity{i}'] = None  # Handle out-of-range output
        except ValueError:
            df.at[index, f'gpt-4o_subjectivity{i}'] = None  # Handle non-integer output

        # Polarity score
        polarity_response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Please provide a single polarity score as a number between -100 and 100 for the sentiment of the following statement, where -100 indicates extremely negative, 0 indicates neutral, and 100 indicates extremely positive. Do not include any additional text or commentary."},
                {"role": "user", "content": f"Statement: {statement}"}
            ]
        )
        
        polarity_score = polarity_response.choices[0].message.content.strip()
        try:
            score = int(polarity_score)
            if -100 <= score <= 100:
                df.at[index, f'gpt-4o_polarity{i}'] = score
            else:
                df.at[index, f'gpt-4o_polarity{i}'] = None  # Handle out-of-range output
        except ValueError:
            df.at[index, f'gpt-4o_polarity{i}'] = None  # Handle non-integer output

# Save the updated DataFrame to a CSV file
output_file_path = 'augmented_data.csv'
df.to_csv(output_file_path, index=False)
