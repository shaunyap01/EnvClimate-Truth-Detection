import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Parameters for text processing and model
MAX_VOCAB_SIZE = 10000  # Maximum number of words in the vocabulary
MAX_SEQUENCE_LENGTH = 100  # Maximum length of text sequences
EMBEDDING_DIM = 100  # Dimension of the embedding layer

# Tokenize the text data and convert it into sequences
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(train_df['statement_clean'])  # Fit tokenizer on training data

X_train_text = pad_sequences(tokenizer.texts_to_sequences(train_df['statement_clean']), maxlen=MAX_SEQUENCE_LENGTH)
X_val_text = pad_sequences(tokenizer.texts_to_sequences(val_df['statement_clean']), maxlen=MAX_SEQUENCE_LENGTH)
X_test_text = pad_sequences(tokenizer.texts_to_sequences(test_df['statement_clean']), maxlen=MAX_SEQUENCE_LENGTH)

# Standardize numerical features (Flesch-Kincaid score, polarity, subjectivity)
scaler = StandardScaler()
X_train_num = scaler.fit_transform(train_df[['flesch_kincaid_ease', 'polarity', 'subjectivity']])
X_val_num = scaler.transform(val_df[['flesch_kincaid_ease', 'polarity', 'subjectivity']])
X_test_num = scaler.transform(test_df[['flesch_kincaid_ease', 'polarity', 'subjectivity']])

# Prepare labels for binary classification
label_mapping = {
    'pants-fire': 0,
    'false': 0,
    'barely-true': 0,
    'half-true': 1,
    'mostly-true': 1,
    'true': 1
}
train_df['target'] = train_df['target'].map(label_mapping)
val_df['target'] = val_df['target'].map(label_mapping)
test_df['target'] = test_df['target'].map(label_mapping)

y_train = train_df['target'].values
y_val = val_df['target'].values
y_test = test_df['target'].values

# Build the CNN model for binary classification
# Text input and embedding layers
text_input = tf.keras.Input(shape=(MAX_SEQUENCE_LENGTH,), name='text_input')
embedding_layer = tf.keras.layers.Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(text_input)
conv_layer = tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu')(embedding_layer)
pooling_layer = tf.keras.layers.GlobalMaxPooling1D()(conv_layer)

# Numerical input and dense layer
numerical_input = tf.keras.Input(shape=(3,), name='numerical_input')
dense_num = tf.keras.layers.Dense(32, activation='relu')(numerical_input)

# Concatenate text and numerical features
concatenated = tf.keras.layers.concatenate([pooling_layer, dense_num])

# Binary output layer with sigmoid activation for classification
output = tf.keras.layers.Dense(1, activation='sigmoid')(concatenated)

# Define the model with inputs and output
model = tf.keras.Model(inputs=[text_input, numerical_input], outputs=output)

# Compile the model with Adam optimizer and binary crossentropy loss
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with training data and validate on validation data
history = model.fit(
    {'text_input': X_train_text, 'numerical_input': X_train_num},
    y_train,
    validation_data=({'text_input': X_val_text, 'numerical_input': X_val_num}, y_val),
    epochs=10,
    batch_size=32
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(
    {'text_input': X_test_text, 'numerical_input': X_test_num}, y_test
)

print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Predict binary labels on the test set
y_pred_prob = model.predict({'text_input': X_test_text, 'numerical_input': X_test_num})
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)  # Convert probabilities to binary predictions

# Generate and display the confusion matrix for model evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)
