import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler

# Define label mapping for binary classification (0: False, 1: True)
label_mapping = {
    'false': 0,
    'barely-true': 0,
    'half-true': 1,
    'mostly-true': 1,
    'true': 1
}

# Apply label mapping to convert target labels into binary labels
train_data['binary_label'] = train_data['target'].map(label_mapping)
val_data['binary_label'] = val_data['target'].map(label_mapping)
test_data['binary_label'] = test_data['target'].map(label_mapping)

# Initialize and fit TF-IDF vectorizer on training data
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['statement_clean'])
X_val_tfidf = tfidf_vectorizer.transform(val_data['statement_clean'])
X_test_tfidf = tfidf_vectorizer.transform(test_data['statement_clean'])

# Stack TF-IDF features with additional numerical features and apply scaling
scaler = StandardScaler(with_mean=False)
X_train = hstack([X_train_tfidf, scaler.fit_transform(train_data[['flesch_kincaid_ease', 'polarity', 'subjectivity']].values)])
X_val = hstack([X_val_tfidf, scaler.transform(val_data[['flesch_kincaid_ease', 'polarity', 'subjectivity']].values)])
X_test = hstack([X_test_tfidf, scaler.transform(test_data[['flesch_kincaid_ease', 'polarity', 'subjectivity']].values)])

# Compute sample weights to handle class imbalance
sample_weights = compute_sample_weight(class_weight='balanced', y=train_data['binary_label'])

# Initialize Random Forest model and define parameter grid for GridSearchCV
rf_model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, train_data['binary_label'], sample_weight=sample_weights)

# Retrieve the best model from Grid Search
best_rf_model = grid_search.best_estimator_

# Predict on the test set using the best model
test_predictions = best_rf_model.predict(X_test)

# Calculate accuracy, precision, recall, and confusion matrix
accuracy = accuracy_score(test_data['binary_label'], test_predictions)
precision = precision_score(test_data['binary_label'], test_predictions, average='binary')
recall = recall_score(test_data['binary_label'], test_predictions, average='binary')
conf_matrix = confusion_matrix(test_data['binary_label'], test_predictions, labels=[0, 1])

# Calculate false positive rate (FPR)
false_positive_rate = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
false_positive_rate = false_positive_rate / conf_matrix.sum(axis=0)

# Display the best parameters and the results
print(f'Best parameters found: {grid_search.best_params_}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'False Positive Rate: {false_positive_rate}')
print('Confusion Matrix:')
print(conf_matrix)
