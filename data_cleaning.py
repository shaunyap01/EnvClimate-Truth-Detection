import pandas as pd
from langdetect import detect, LangDetectException
import unicodedata

# Function to detect if a given text is in English
def is_english(text):
    try:
        return detect(text) == 'en'  # Return True if the text is detected as English
    except LangDetectException:  # Handle exceptions if language detection fails
        return False

# Assuming 'data' is your DataFrame, filter rows to keep only English statements
data = data[data['statement'].apply(is_english)]

# Normalize and clean the 'statement' text
# Convert any unicode characters to their closest ASCII representation
data['statement_clean'] = data['statement'].apply(
    lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8')
)

# Convert the 'date' column to datetime format for easier manipulation and analysis
data['date'] = pd.to_datetime(data['date'], format='%B %d %Y')

# Remove rows where the 'target' column has irrelevant values like 'full-flop', 'half-flip', or 'no-flip'
data = data[~data['target'].isin(['full-flop', 'half-flip', 'no-flip'])]

# Keep only the cleaned 'statement_clean' and 'target' columns in the final DataFrame
data = data[['statement_clean', 'target']]

# Optional: Reset the index of the DataFrame after filtering
data.reset_index(drop=True, inplace=True)

# Display the first few rows of the cleaned DataFrame
print(data.head())
