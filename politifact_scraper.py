# Import necessary libraries
from bs4 import BeautifulSoup
import pandas as pd
import requests

# Initialize lists to store the scraped data
authors = []
dates = []
statements = []
sources = []
targets = []

# Function to scrape data from a specific category on PolitiFact
def scrape_website(category, page_count):
    for page_number in range(1, page_count + 1):
        # Convert the page number to a string
        page_num = str(page_number)
        
        # Construct the URL based on the category and page number
        URL = f'https://www.politifact.com/factchecks/list/?page={page_num}&category={category}'
        
        # Make a request to the website
        webpage = requests.get(URL)
        soup = BeautifulSoup(webpage.text, "html.parser")
        
        # Extract the relevant data from the HTML
        statement_footer = soup.find_all('footer', attrs={'class': 'm-statement__footer'})
        statement_quote = soup.find_all('div', attrs={'class': 'm-statement__quote'})
        statement_meta = soup.find_all('div', attrs={'class': 'm-statement__meta'})
        target = soup.find_all('div', attrs={'class': 'm-statement__meter'})
        
        # Loop through the footer class to extract the author and date
        for footer in statement_footer:
            text = footer.text.strip().split()
            # Extract the date (last three elements)
            date = ' '.join(text[-3:])
            dates.append(date)
            # Extract the author name (excluding titles like 'By')
            author = ' '.join(text[1:-4])
            authors.append(author)
        
        # Loop through the quote class to extract the statement text
        for quote in statement_quote:
            statement = quote.find('a').text.strip()
            statements.append(statement)
        
        # Loop through the meta class to extract the source
        for meta in statement_meta:
            source = meta.find('a').text.strip()
            sources.append(source)
        
        # Loop through the meter class to extract the truthfulness rating
        for meter in target:
            truth_rating = meter.find('div', attrs={'class': 'c-image'}).find('img')['alt']
            targets.append(truth_rating)

# Scrape data from the "environment" category (23 pages)
scrape_website('environment', 23)

# Scrape data from the "climate-change" category (13 pages)
scrape_website('climate-change', 13)

# Create a DataFrame to store the scraped data
data = pd.DataFrame({
    'author': authors,
    'statement': statements,
    'source': sources,
    'date': dates,
    'target': targets
})

# Display the first few rows of the DataFrame
print(data.head())

# Save the DataFrame to a CSV file for later use
data.to_csv('politifact_climate_environment_data.csv', index=False)
