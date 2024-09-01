import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from scipy.sparse import hstack
from imblearn.over_sampling import RandomOverSampler

# Vectorize the textual data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['statement_clean'])
X_val_tfidf = tfidf_vectorizer.transform(val_data['statement_clean'])
X_test_tfidf = tfidf_vectorizer.transform(test_data['statement_clean'])

# Combine TF-IDF features with additional numerical features
X_train = hstack([X_train_tfidf, train_data[['flesch_kincaid_ease', 'polarity', 'subjectivity']].values])
X_val = hstack([X_val_tfidf, val_data[['flesch_kincaid_ease', 'polarity', 'subjectivity']].values])
X_test = hstack([X_test_tfidf, test_data[['flesch_kincaid_ease', 'polarity', 'subjectivity']].values])

# Apply ordinal encoding to the labels
# Define the order for ordinal encoding
categories = [['false', 'barely-true', 'half-true', 'mostly-true', 'true']]
ordinal_encoder = OrdinalEncoder(categories=categories)
y_train = ordinal_encoder.fit_transform(train_data[['target']]).ravel()
y_val = ordinal_encoder.transform(val_data[['target']]).ravel()
y_test = ordinal_encoder.transform(test_data[['target']]).ravel()

# Address class imbalance using Random Over-Sampling
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Define hyperparameter grid for SVM tuning
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
    'degree': [2, 3, 4],  # Relevant for 'poly' kernel
}

# Initialize and tune the SVM model using GridSearchCV
svm_model = SVC()
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)

# Retrieve the best model from GridSearchCV
best_svm_model = grid_search.best_estimator_

# Predict on the test set using the best SVM model
test_predictions = best_svm_model.predict(X_test)

# Decode predicted labels back to their original categories
predicted_labels = ordinal_encoder.inverse_transform(test_predictions.reshape(-1, 1)).ravel()

# Evaluate model performance: accuracy and confusion matrix
accuracy = accuracy_score(test_data['target'], predicted_labels)
conf_matrix = confusion_matrix(test_data['target'], predicted_labels, labels=['false', 'barely-true', 'half-true', 'mostly-true', 'true'])

# Display the best hyperparameters and results
print(f'Best parameters found: {grid_search.best_params_}')
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
