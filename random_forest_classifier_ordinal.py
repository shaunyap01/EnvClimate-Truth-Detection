import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from scipy.sparse import hstack
from imblearn.over_sampling import RandomOverSampler

# Initialize and fit TF-IDF vectorizer on training data
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['statement_clean'])
X_val_tfidf = tfidf_vectorizer.transform(val_data['statement_clean'])
X_test_tfidf = tfidf_vectorizer.transform(test_data['statement_clean'])

# Stack TF-IDF features with additional numerical features
X_train = hstack([X_train_tfidf, train_data[['flesch_kincaid_ease', 'polarity', 'subjectivity']].values])
X_val = hstack([X_val_tfidf, val_data[['flesch_kincaid_ease', 'polarity', 'subjectivity']].values])
X_test = hstack([X_test_tfidf, test_data[['flesch_kincaid_ease', 'polarity', 'subjectivity']].values])

# Ordinal encoding of the target labels
# Define the order for ordinal encoding
categories = [['false', 'barely-true', 'half-true', 'mostly-true', 'true']]
ordinal_encoder = OrdinalEncoder(categories=categories)
y_train = ordinal_encoder.fit_transform(train_data[['target']]).ravel()
y_val = ordinal_encoder.transform(val_data[['target']]).ravel()
y_test = ordinal_encoder.transform(test_data[['target']]).ravel()

# Balance the classes using Random Over-Sampling
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

# Initialize the Random Forest model and perform Grid Search
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)

# Retrieve the best model from Grid Search
best_rf_model = grid_search.best_estimator_

# Predict on the test set using the best model
test_predictions = best_rf_model.predict(X_test)

# Decode the predicted labels back to the original categories
predicted_labels = ordinal_encoder.inverse_transform(test_predictions.reshape(-1, 1)).ravel()

# Display the best parameters and the results
print(f'Best parameters found: {grid_search.best_params_}')
print('Confusion Matrix:')
print(conf_matrix)
