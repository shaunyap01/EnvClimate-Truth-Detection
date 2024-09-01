import textstat
from textblob import TextBlob
import pandas as pd

# Function to calculate the Flesch Reading Ease score
def calculate_flesch_reading_ease(text):
    ease = textstat.flesch_reading_ease(text)
    return ease

# Function to calculate polarity and subjectivity using TextBlob
def calculate_polarity_subjectivity(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # Ranges from -1 (negative) to 1 (positive)
    subjectivity = blob.sentiment.subjectivity  # Ranges from 0 (objective) to 1 (subjective)
    return polarity, subjectivity

# Assuming df is your DataFrame that contains the cleaned statements
# Adding new features to the DataFrame
df['flesch_kincaid_ease'] = df['statement_clean'].apply(calculate_flesch_reading_ease)
df[['polarity', 'subjectivity']] = df['statement_clean'].apply(lambda x: pd.Series(calculate_polarity_subjectivity(x)))

# Display the first few rows of the updated DataFrame to verify the new features
print(df.head())

# Save the updated DataFrame with new features to a CSV file for later use
df.to_csv('preprocessed_politifact_data.csv', index=False)
