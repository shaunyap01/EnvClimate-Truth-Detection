import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.sparse import hstack

# Define label mapping for ordinal classification
label_mapping = {
    'false': 0,
    'barely-true': 0.25,
    'half-true': 0.5,
    'mostly-true': 0.75,
    'true': 1
}

# Apply label mapping to train, validation, and test datasets
train_data['label'] = train_data['target'].map(label_mapping)
val_data['label'] = val_data['target'].map(label_mapping)
test_data['label'] = test_data['target'].map(label_mapping)

# Initialize and fit TF-IDF vectorizer on training data
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['statement_clean'])
X_val_tfidf = tfidf_vectorizer.transform(val_data['statement_clean'])
X_test_tfidf = tfidf_vectorizer.transform(test_data['statement_clean'])

# Stack TF-IDF features with additional numerical features (readability, polarity, subjectivity)
X_train = hstack([X_train_tfidf, train_data[['flesch_kincaid_ease', 'polarity', 'subjectivity']].values])
X_val = hstack([X_val_tfidf, val_data[['flesch_kincaid_ease', 'polarity', 'subjectivity']].values])
X_test = hstack([X_test_tfidf, test_data[['flesch_kincaid_ease', 'polarity', 'subjectivity']].values])

# Train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, train_data['label'])

# Predict on the test set
test_predictions = lr_model.predict(X_test)

# Function to map continuous predictions back to ordinal categories
def bin_labels(value):
    if value < 0.125:
        return 'false'  # Combine 'pants-fire' and 'false'
    elif value < 0.375:
        return 'barely-true'
    elif value < 0.625:
        return 'half-true'
    elif value < 0.875:
        return 'mostly-true'
    else:
        return 'true'

# Apply the binning function to the predictions to obtain final labels
predicted_labels = np.array([bin_labels(pred) for pred in test_predictions])

# Calculate accuracy of the model
accuracy = accuracy_score(test_data['target'], predicted_labels)

# Generate the confusion matrix for detailed performance analysis
conf_matrix = confusion_matrix(test_data['target'], predicted_labels, labels=['false', 'barely-true', 'half-true', 'mostly-true', 'true'])

# Display the results
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
