import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Download required NLTK data
#nltk.download('punkt', quiet=True)
#@nltk.download('wordnet', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents.json
intents = json.loads(open('intents.json').read())

# Initialize lists
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',', ';', ':']

# Preprocess the data
print("Preprocessing data...")
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word_list = nltk.word_tokenize(pattern)
        # Add to words list
        words.extend(word_list)
        # Add to documents with tag
        documents.append((word_list, intent["tag"]))
        # Add to classes if new
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize and clean words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

print(f"Unique words: {len(words)}")
print(f"Classes: {classes}")
print(f"Total documents: {len(documents)}")

# Save words and classes
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Create training data
training_features = []
training_labels = []

# Create an output empty array for each class
output_empty = [0] * len(classes)

print("\nCreating bag of words...")
for document in documents:
    bag = []
    # Get list of tokenized words for the pattern
    pattern_words = document[0]
    # Lemmatize each word
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # Create bag of words
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    
    # Create output row
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    
    training_features.append(bag)
    training_labels.append(output_row)

# Convert to numpy arrays

train_x = np.array(training_features)
train_y = np.array(training_labels)

print(f"\nTraining data shape: X={train_x.shape}, Y={train_y.shape}")
print(f"Number of training samples: {len(train_x)}")

y_integers = np.argmax(train_y, axis=1)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_integers),
    y=y_integers
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.20, random_state=42, stratify=train_y)
print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")

# Build a better model
print("\nBuilding model...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

     tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])

# Compile with better optimizer
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True,
    verbose=1,
    min_delta=0.001
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=15,
    min_lr=0.00001,
    verbose=1
)

# Train the model
print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=8,
    verbose=1,
    callbacks=[early_stopping, reduce_lr]
)

# Save the model
model.save("chatbot2(mod)_model.h5")
print('\nModel trained and saved successfully!')

# Evaluate on full dataset

print("\nEvaluation on full dataset:")
loss, accuracy = model.evaluate(train_x, train_y, verbose=0)
print(f"Training Accuracy: {accuracy:.4f}")
print(f"Training Loss: {loss:.4f}")
"""
# Test predictions
print("\nTest predictions:")
test_samples = [
    "hello there",
    "goodbye friend",
    "thank you so much",
    "what is simplilearn?"
]

for test_sample in test_samples:
    # Create bag of words for test sample
    bag = [0] * len(words)
    test_words = nltk.word_tokenize(test_sample)
    test_words = [lemmatizer.lemmatize(word.lower()) for word in test_words]
    
    for word in test_words:
        if word in words:
            bag[words.index(word)] = 1
    
    # Predict
    prediction = model.predict(np.array([bag]), verbose=0)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    print(f"Input: '{test_sample}'")
    print(f"  Predicted: {predicted_class} (confidence: {confidence:.4f})")
    """