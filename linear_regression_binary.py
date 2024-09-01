import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.sparse import hstack

# Define label mapping for binary classification
label_mapping = {
    'false': 0,
    'barely-true': 0,
    'half-true': 1,
    'mostly-true': 1,
    'true': 1
}

# Map the labels in the dataset to binary values
train_data['binary_label'] = train_data['target'].map(label_mapping)
val_data['binary_label'] = val_data['target'].map(label_mapping)
test_data['binary_label'] = test_data['target'].map(label_mapping)

# Define TfidfVectorizer to convert text data into numerical features
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform the text data into TF-IDF features
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['statement_clean'])
X_val_tfidf = tfidf_vectorizer.transform(val_data['statement_clean'])
X_test_tfidf = tfidf_vectorizer.transform(test_data['statement_clean'])

# Combine TF-IDF features with additional numerical features
X_train = hstack([X_train_tfidf, train_data[['flesch_kincaid_ease', 'polarity', 'subjectivity']].values])
X_val = hstack([X_val_tfidf, val_data[['flesch_kincaid_ease', 'polarity', 'subjectivity']].values])
X_test = hstack([X_test_tfidf, test_data[['flesch_kincaid_ease', 'polarity', 'subjectivity']].values])

# Train a Linear Regression model on the training data
lr_model = LinearRegression()
lr_model.fit(X_train, train_data['binary_label'])

# Predict binary labels on the test set
test_predictions = lr_model.predict(X_test)

# Function to convert regression outputs to binary labels
def bin_labels(value):
    return 1 if value >= 0.5 else 0

# Apply the binning function to the model predictions
predicted_labels = np.array([bin_labels(pred) for pred in test_predictions])

# Calculate the accuracy of the predictions
accuracy = accuracy_score(test_data['binary_label'], predicted_labels)

# Generate the confusion matrix to evaluate the model's performance
conf_matrix = confusion_matrix(test_data['binary_label'], predicted_labels, labels=[0, 1])

# Display the accuracy and confusion matrix
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
