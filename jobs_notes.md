# **Complete Explanation of jobs.py - Chatbot Training Script**

## **Overview**
`jobs.py` is the core training script that handles:
- Data preprocessing and augmentation
- Neural network model creation
- Training with advanced techniques
- Model evaluation and saving

Let me break down each section in detail:

---

## **1. Imports and Setup**

```python
# Core imports
import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, Embedding, Dropout, Bidirectional, 
    SpatialDropout1D, Conv1D, GlobalMaxPooling1D,
    BatchNormalization, MultiHeadAttention, LayerNormalization,
    Input, concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard, LearningRateScheduler
)

# Data processing imports
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Utility imports
import json
import pickle
import re
import random
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from typing import List, Tuple, Dict
import joblib

# NLTK downloads
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
```

**Why these imports?**
- **TensorFlow/Keras**: Deep learning framework
- **NLTK**: Natural Language Processing toolkit
- **Scikit-learn**: Machine learning utilities
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Visualization

---

## **2. Enhanced Text Preprocessor Class**

```python
class EnhancedTextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.vocab = set()
        self.word2idx = {}
        self.idx2word = {}
        self.max_sequence_length = 0
        self.tfidf_vectorizer = None
        self.word2vec_model = None
```

### **Text Cleaning Method**
```python
def clean_text(self, text):
    """Advanced text cleaning and normalization"""
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits but keep basic punctuation
    text = re.sub(r"[^a-zA-Z\s\?\!\.\,]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
```

**What this does:**
- `text.lower()`: Converts "Hello" to "hello" for consistency
- `re.sub(r"[^a-zA-Z\s\?\!\.\,]", "", text)`: Removes numbers and special chars but keeps punctuation
- `re.sub(r"\s+", " ", text).strip()`: Replaces multiple spaces with single space

**Example:**
- Input: "Hello!  How are you?? 123"
- Output: "hello! how are you??"

### **Advanced Tokenization**
```python
def advanced_tokenize(self, text):
    """Advanced tokenization with POS tagging"""
    if not text:
        return []
        
    tokens = word_tokenize(text)

    # Remove stopwords and short tokens
    tokens = [
        token for token in tokens if token not in self.stop_words and len(token) > 1
    ]

    # Lemmatization with POS tagging
    try:
        pos_tags = nltk.pos_tag(tokens)
        lemmatized_tokens = []

        for token, pos in pos_tags:
            if pos.startswith("V"):  # Verb
                lemma = self.lemmatizer.lemmatize(token, pos="v")
            elif pos.startswith("J"):  # Adjective
                lemma = self.lemmatizer.lemmatize(token, pos="a")
            elif pos.startswith("R"):  # Adverb
                lemma = self.lemmatizer.lemmatize(token, pos="r")
            else:  # Noun and others
                lemma = self.lemmatizer.lemmatize(token)
            lemmatized_tokens.append(lemma)

        return lemmatized_tokens
    except:
        return [self.lemmatizer.lemmatize(token) for token in tokens]
```

**Step-by-step process:**
1. **Tokenization**: `"hello how are you"` → `["hello", "how", "are", "you"]`
2. **Stopword removal**: Removes common words like "how", "are", "you"
3. **POS Tagging**: Identifies parts of speech `[("hello", "NN"), ("how", "WRB")]`
4. **Lemmatization**: Converts to base form based on POS

**Example:**
- Input: "I am running quickly to the better store"
- Output: `["I", "run", "quick", "good", "store"]`

### **Vocabulary Building**
```python
def build_vocabulary(self, documents):
    """Build vocabulary from all documents"""
    all_tokens = []
    for doc in documents:
        cleaned_text = self.clean_text(doc)
        tokens = self.advanced_tokenize(cleaned_text)
        all_tokens.extend(tokens)
        self.vocab.update(tokens)

    # Create word to index mapping
    self.vocab = sorted(self.vocab)
    self.word2idx = {
        word: idx + 1 for idx, word in enumerate(self.vocab)
    }  # 0 reserved for padding
    self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    # Calculate max sequence length
    sequence_lengths = [
        len(self.advanced_tokenize(self.clean_text(doc))) for doc in documents
    ]
    if sequence_lengths:
        self.max_sequence_length = max(
            5, int(np.percentile(sequence_lengths, 95))
        )
    else:
        self.max_sequence_length = 20

    return all_tokens
```

**What happens here:**
1. **Collect all tokens** from all documents
2. **Create mapping**: word → index (starting from 1, 0 for padding)
3. **Calculate max sequence length** using 95th percentile to handle outliers

**Example vocabulary:**
```python
word2idx = {
    "hello": 1,
    "book": 2, 
    "table": 3,
    "menu": 4,
    # ...
}
```

### **Text to Sequence Conversion**
```python
def text_to_sequence(self, text):
    """Convert text to sequence of indices"""
    cleaned_text = self.clean_text(text)
    tokens = self.advanced_tokenize(cleaned_text)
    sequence = [self.word2idx.get(token, 0) for token in tokens]  # 0 for OOV
    return sequence
```

**Example:**
- Input: "hello book table"
- Process: `["hello", "book", "table"]`
- Output: `[1, 2, 3]` (indices from vocabulary)

---

## **3. Data Enhancement Strategies**

### **Data Augmentation**
```python
def augment_training_data(documents: List[str], labels: List[str], augmentation_factor: int = 2):
    """Augment training data using various techniques"""
    augmented_docs = documents.copy()
    augmented_labels = labels.copy()
    
    for doc, label in zip(documents, labels):
        # Synonym replacement
        try:
            augmented_docs.append(synonym_replacement(doc))
            augmented_labels.append(label)
        except:
            pass
            
        # Random insertion
        try:
            augmented_docs.append(random_insertion(doc))
            augmented_labels.append(label)
        except:
            pass
            
        # Back translation simulation (simple paraphrasing)
        try:
            augmented_docs.append(paraphrase_text(doc))
            augmented_labels.append(label)
        except:
            pass
    
    return augmented_docs, augmented_labels
```

**Why augment data?**
- Creates more training examples
- Helps model generalize better
- Handles variations in user input

### **Synonym Replacement**
```python
def get_synonyms(word: str) -> List[str]:
    """Get synonyms for a word using WordNet"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            if synonym != word and len(synonym.split()) == 1:
                synonyms.add(synonym)
    return list(synonyms)

def synonym_replacement(text: str, n: int = 1) -> str:
    """Replace words with synonyms"""
    words = text.split()
    new_words = words.copy()
    random_word_list = [word for word in words if word not in stopwords.words('english')]
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    
    return ' '.join(new_words)
```

**Example:**
- Input: "I want to book a table"
- Output: "I wish to reserve a table"

### **Data Balancing**
```python
def balance_dataset(documents: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    """Balance the dataset across classes"""
    label_counts = Counter(labels)
    max_count = max(label_counts.values())
    
    balanced_docs = []
    balanced_labels = []
    
    for label in set(labels):
        label_docs = [doc for doc, lbl in zip(documents, labels) if lbl == label]
        
        if len(label_docs) < max_count:
            # Upsample minority classes
            upsampled_docs = resample(
                label_docs, 
                replace=True, 
                n_samples=max_count, 
                random_state=42
            )
            balanced_docs.extend(upsampled_docs)
            balanced_labels.extend([label] * max_count)
        else:
            balanced_docs.extend(label_docs)
            balanced_labels.extend([label] * len(label_docs))
    
    return balanced_docs, balanced_labels
```

**Why balance data?**
- Prevents model from favoring frequent classes
- Improves performance on all intents equally

---

## **4. Advanced Model Architectures**

### **Main Advanced Model**
```python
def create_advanced_model(vocab_size: int, num_classes: int, max_sequence_length: int, 
                         embedding_dim: int = 300, dropout_rate: float = 0.3, learning_rate: float = 0.001):
    """Create a more sophisticated model architecture"""
    
    model = Sequential([
        # 1. Embedding Layer - Convert words to vectors
        Embedding(
            input_dim=vocab_size + 1,  # +1 for padding
            output_dim=embedding_dim,   # 300-dimensional vectors
            input_length=max_sequence_length,
            mask_zero=True,  # Ignore padding
        ),
        
        # 2. Spatial Dropout - Prevent overfitting
        SpatialDropout1D(dropout_rate),
        
        # 3. Multi-scale CNN - Extract local patterns
        Conv1D(64, 2, activation='relu', padding='same'),  # Bigrams
        Conv1D(64, 3, activation='relu', padding='same'),  # Trigrams  
        Conv1D(64, 4, activation='relu', padding='same'),  # 4-grams
        
        # 4. Bidirectional LSTM - Understand context
        Bidirectional(LSTM(
            128, 
            return_sequences=True, 
            dropout=0.2, 
            recurrent_dropout=0.2,
            kernel_regularizer=l2(0.01)  # Prevent overfitting
        )),
        
        # 5. Global Pooling - Reduce to fixed size
        GlobalMaxPooling1D(),
        
        # 6. Dense Layers - Learn complex patterns
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),  # Stabilize training
        Dropout(0.5),          # Prevent overfitting
        
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        # 7. Output Layer - Predict intent probabilities
        Dense(num_classes, activation='softmax')
    ])
    
    # Optimizer with learning rate and gradient clipping
    optimizer = Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0  # Prevent exploding gradients
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model
```

### **Layer Explanation:**

1. **Embedding Layer**: 
   - Converts word indices to dense vectors
   - `mask_zero=True` ignores padding (0s)

2. **Spatial Dropout**:
   - Drops entire feature maps instead of individual elements
   - Better for convolutional layers

3. **Multi-scale CNN**:
   - **Conv1D(64, 2)**: Learns 2-word phrases ("book table")
   - **Conv1D(64, 3)**: Learns 3-word phrases ("want to book")  
   - **Conv1D(64, 4)**: Learns 4-word phrases

4. **Bidirectional LSTM**:
   - Reads text both forward and backward
   - Understands context from both directions

5. **Global Max Pooling**:
   - Takes maximum value from each feature map
   - Converts variable-length sequences to fixed size

6. **Dense Layers**:
   - Learn complex patterns from extracted features
   - BatchNorm stabilizes training
   - Dropout prevents overfitting

7. **Output Layer**:
   - Softmax gives probability distribution over intents

### **Alternative Model Architectures**

```python
def create_transformer_model(vocab_size: int, num_classes: int, max_sequence_length: int, embedding_dim: int = 128):
    """Transformer-based model for better context understanding"""
    
    inputs = Input(shape=(max_sequence_length,))
    
    # Embedding layer
    embedding = Embedding(vocab_size + 1, embedding_dim)(inputs)
    embedding = SpatialDropout1D(0.2)(embedding)
    
    # Transformer block with self-attention
    attention_output = MultiHeadAttention(
        num_heads=8, 
        key_dim=embedding_dim // 8
    )(embedding, embedding)
    
    attention_output = Dropout(0.1)(attention_output)
    attention_output = LayerNormalization()(embedding + attention_output)
    
    # Feed forward network
    ffn_output = Dense(512, activation='relu')(attention_output)
    ffn_output = Dense(embedding_dim)(ffn_output)
    ffn_output = Dropout(0.1)(ffn_output)
    ffn_output = LayerNormalization()(attention_output + ffn_output)
    
    # Global pooling and output
    pooled = GlobalMaxPooling1D()(ffn_output)
    dense = Dense(128, activation='relu')(pooled)
    dense = Dropout(0.3)(dense)
    outputs = Dense(num_classes, activation='softmax')(dense)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

---

## **5. Advanced Training Techniques**

```python
def train_advanced_model(model, X_train, y_train, X_val, y_val, epochs: int = 200):
    """Enhanced training with advanced techniques"""
    
    def lr_scheduler(epoch, lr):
        if epoch < 10:
            return float(lr)  # Keep initial learning rate
        else:
            return float(lr * tf.math.exp(-0.1))  # Exponential decay
    
    callbacks = [
        # Stop when validation loss stops improving
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            min_delta=0.001,
            verbose=1
        ),
        
        # Reduce learning rate when stuck
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,      # Halve the learning rate
            patience=10,     # Wait 10 epochs
            min_lr=1e-7,     # Minimum learning rate
            verbose=1
        ),
        
        # Save the best model
        ModelCheckpoint(
            'filesTech/best_model_advanced.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Learning rate scheduling
        LearningRateScheduler(lr_scheduler)
    ]
    
    # Handle class imbalance
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(np.argmax(y_train, axis=1)),
        y=np.argmax(y_train, axis=1)
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight_dict,  # Weight imbalanced classes
        verbose=1,
        shuffle=True
    )
    
    return history
```

**Training Callbacks Explained:**

1. **EarlyStopping**: 
   - Stops training when validation loss stops improving
   - `patience=20`: Wait 20 epochs before stopping
   - `restore_best_weights=True`: Keep the best model weights

2. **ReduceLROnPlateau**:
   - Reduces learning rate when model gets stuck
   - `factor=0.5`: Halve the learning rate
   - Helps model escape local minima

3. **ModelCheckpoint**:
   - Saves the best model automatically
   - Prevents losing good models during training

4. **LearningRateScheduler**:
   - Gradually decreases learning rate over time
   - Helps with fine-tuning in later epochs

---

## **6. Ensemble Methods**

```python
class ModelEnsemble:
    def __init__(self, vocab_size: int, num_classes: int, max_sequence_length: int):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.max_sequence_length = max_sequence_length
        self.models = {}
        
    def create_ensemble(self):
        """Create ensemble of multiple models"""
        self.models = {
            'advanced_model': create_advanced_model(
                self.vocab_size, self.num_classes, self.max_sequence_length
            ),
            'transformer_model': create_transformer_model(
                self.vocab_size, self.num_classes, self.max_sequence_length
            ),
            'cnn_model': create_cnn_model(
                self.vocab_size, self.num_classes, self.max_sequence_length
            )
        }
        
    def train_ensemble(self, X_train, y_train, X_val, y_val, epochs: int = 100):
        """Train all models in the ensemble"""
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                verbose=0,  # Quiet training
                callbacks=[
                    EarlyStopping(patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(patience=5)
                ]
            )
            
    def predict_ensemble(self, X):
        """Combine predictions from multiple models"""
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        
        # Weighted average based on expected performance
        weights = [0.4, 0.4, 0.2]  # Advanced and Transformer get higher weights
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        return weighted_pred
```

**Why Ensemble?**
- **Diversity**: Different architectures capture different patterns
- **Robustness**: Less likely to all make the same mistake
- **Performance**: Combined predictions often beat single models

---

## **7. Hyperparameter Optimization**

```python
def manual_hyperparameter_tuning(X_train, y_train, X_val, y_val, vocab_size: int, num_classes: int, max_sequence_length: int):
    """Manual hyperparameter tuning with predefined configurations"""
    
    configurations = [
        {
            'embedding_dim': 100,
            'learning_rate': 0.001,
            'dropout_rate': 0.3,
            'description': 'Baseline Configuration'
        },
        {
            'embedding_dim': 200,
            'learning_rate': 0.0005,
            'dropout_rate': 0.4,
            'description': 'Higher Dimension Configuration'
        },
        {
            'embedding_dim': 300,
            'learning_rate': 0.001,
            'dropout_rate': 0.5,
            'description': 'Larger Model Configuration'
        },
        {
            'embedding_dim': 150,
            'learning_rate': 0.0001,
            'dropout_rate': 0.2,
            'description': 'Conservative Configuration'
        }
    ]
    
    best_accuracy = 0
    best_config = None
    best_model = None
    
    for i, config in enumerate(configurations):
        print(f"\nTesting Configuration {i+1}: {config['description']}")
        
        model = create_advanced_model(
            vocab_size=vocab_size,
            num_classes=num_classes,
            max_sequence_length=max_sequence_length,
            embedding_dim=config['embedding_dim'],
            dropout_rate=config['dropout_rate'],
            learning_rate=config['learning_rate']
        )
        
        # Quick training for evaluation
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,  # Fewer epochs for quick evaluation
            batch_size=32,
            verbose=0,
            callbacks=[
                EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy')
            ]
        )
        
        val_accuracy = max(history.history['val_accuracy'])
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_config = config
            best_model = model
    
    print(f"\nBest Configuration: {best_config['description']}")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    
    return best_config, best_model
```

**Hyperparameters Explained:**
- **embedding_dim**: Size of word vectors (higher = more expressive but slower)
- **learning_rate**: Step size for weight updates (smaller = more precise but slower)
- **dropout_rate**: Percentage of neurons to ignore during training (prevents overfitting)

---

## **8. Main Training Function**

```python
def my_jobs():
    print("Starting advanced NLP preprocessing and model training with all improvements...")

    # 1. Load training data
    with open("filesTech/final.json", encoding="utf-8") as file:
        data = json.load(file)

    # Extract patterns and labels
    documents = []
    labels = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            documents.append(pattern)
            labels.append(intent["tag"])

    # 2. Data augmentation and balancing
    print("1. Applying data augmentation...")
    augmented_docs, augmented_labels = augment_training_data(documents, labels)
    
    print("2. Balancing dataset...")
    balanced_docs, balanced_labels = balance_dataset(augmented_docs, augmented_labels)
    
    print(f"Original data: {len(documents)} samples")
    print(f"After augmentation: {len(augmented_docs)} samples")
    print(f"After balancing: {len(balanced_docs)} samples")

    # 3. Text preprocessing
    preprocessor = EnhancedTextPreprocessor()
    print("3. Building vocabulary and extracting features...")
    preprocessor.build_vocabulary(balanced_docs)

    # Convert text to sequences
    sequences = [preprocessor.text_to_sequence(doc) for doc in balanced_docs]
    
    # Pad sequences to same length
    X = pad_sequences(
        sequences,
        maxlen=preprocessor.max_sequence_length,
        padding="post",    # Add zeros at the end
        truncating="post"  # Cut from the end if too long
    )

    # Prepare labels
    unique_labels = sorted(set(balanced_labels))
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label2idx[label] for label in balanced_labels])

    # Convert labels to categorical (one-hot encoding)
    y_categorical = tf.keras.utils.to_categorical(y, num_classes=len(unique_labels))

    # 4. Split data into train/validation/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_categorical, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=[np.argmax(arr) for arr in y_temp],
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Vocabulary size: {len(preprocessor.vocab)}")
    print(f"Max sequence length: {preprocessor.max_sequence_length}")

    # 5. Hyperparameter tuning
    print("4. Starting manual hyperparameter tuning...")
    best_config, tuned_model = manual_hyperparameter_tuning(
        X_train, y_train, X_val, y_val, 
        len(preprocessor.vocab), len(unique_labels), preprocessor.max_sequence_length
    )

    # 6. Train final model with best configuration
    print("5. Creating and training advanced model with best configuration...")
    advanced_model = create_advanced_model(
        vocab_size=len(preprocessor.vocab),
        num_classes=len(unique_labels),
        max_sequence_length=preprocessor.max_sequence_length,
        embedding_dim=best_config['embedding_dim'],
        dropout_rate=best_config['dropout_rate'],
        learning_rate=best_config['learning_rate']
    )

    print("Advanced Model architecture:")
    advanced_model.summary()

    # Train the model
    print("Training advanced model...")
    history = train_advanced_model(advanced_model, X_train, y_train, X_val, y_val, epochs=150)

    # 7. Create and train ensemble
    print("6. Creating model ensemble...")
    ensemble = ModelEnsemble(
        vocab_size=len(preprocessor.vocab),
        num_classes=len(unique_labels),
        max_sequence_length=preprocessor.max_sequence_length
    )
    ensemble.create_ensemble()
    ensemble.train_ensemble(X_train, y_train, X_val, y_val, epochs=80)

    # 8. Comprehensive evaluation
    print("7. Comprehensive evaluation...")
    
    # Evaluate single model
    print("\n=== SINGLE MODEL EVALUATION ===")
    single_accuracy, single_report, single_cm = comprehensive_evaluation(
        advanced_model, X_test, y_test, unique_labels
    )
    
    # Evaluate ensemble
    print("\n=== ENSEMBLE MODEL EVALUATION ===")
    ensemble_pred = ensemble.predict_ensemble(X_test)
    ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    ensemble_accuracy = accuracy_score(y_true_classes, ensemble_pred_classes)
    ensemble_report = classification_report(y_true_classes, ensemble_pred_classes, target_names=unique_labels)
    
    print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
    print("\nEnsemble Classification Report:")
    print(ensemble_report)

    # 9. Save everything
    print("8. Saving models and artifacts...")
    
    # Save single model
    advanced_model.save("filesTech/advanced_model_improved.h5")
    
    # Save ensemble
    ensemble.save_ensemble("filesTech/ensemble_models")
    
    # Save preprocessing artifacts for later use
    with open("filesTech/enhanced_preprocessor.pickle", "wb") as f:
        pickle.dump(
            {
                "preprocessor": preprocessor,
                "label2idx": label2idx,
                "idx2label": {idx: label for label, idx in label2idx.items()},
                "max_sequence_length": preprocessor.max_sequence_length,
                "vocab_size": len(preprocessor.vocab),
                "best_config": best_config
            },
            f,
        )

    # Plot training history
    if len(history.history["accuracy"]) > 1:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.savefig("filesTech/improved_training_history.png")
        plt.close()

    print("\n=== TRAINING SUMMARY ===")
    print(f"Single Model Accuracy: {single_accuracy:.4f}")
    print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
    print(f"Improvement: {((ensemble_accuracy - single_accuracy) / single_accuracy * 100):.2f}%")
    print(f"Best Configuration: {best_config['description']}")
    print("Training completed successfully!")
```

---

## **9. Prediction Function**

```python
def predict_intent_improved(
    text,
    model_path="filesTech/advanced_model_improved.h5",
    preprocessor_path="filesTech/enhanced_preprocessor.pickle",
    use_ensemble: bool = False
):
    """Predict intent using improved model"""
    try:
        # Load preprocessor and model
        with open(preprocessor_path, "rb") as f:
            artifacts = pickle.load(f)

        preprocessor = artifacts["preprocessor"]
        label2idx = artifacts["label2idx"]
        idx2label = artifacts["idx2label"]

        if use_ensemble:
            # Load ensemble
            ensemble = ModelEnsemble(
                vocab_size=artifacts["vocab_size"],
                num_classes=len(label2idx),
                max_sequence_length=artifacts["max_sequence_length"]
            )
            ensemble.load_ensemble("filesTech/ensemble_models")
            
            # Preprocess and predict
            sequence = preprocessor.text_to_sequence(text)
            padded_sequence = pad_sequences(
                [sequence],
                maxlen=preprocessor.max_sequence_length,
                padding="post",
                truncating="post",
            )
            
            prediction = ensemble.predict_ensemble(padded_sequence)[0]
        else:
            # Load single model
            model = tf.keras.models.load_model(model_path)
            
            # Preprocess and predict
            sequence = preprocessor.text_to_sequence(text)
            padded_sequence = pad_sequences(
                [sequence],
                maxlen=preprocessor.max_sequence_length,
                padding="post",
                truncating="post",
            )
            
            prediction = model.predict(padded_sequence, verbose=0)[0]

        predicted_idx = np.argmax(prediction)
        confidence = prediction[predicted_idx]

        return {
            "intent": idx2label[predicted_idx],
            "confidence": float(confidence),
            "all_probabilities": {
                idx2label[i]: float(prob) for i, prob in enumerate(prediction)
            },
            "model_type": "ensemble" if use_ensemble else "single"
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        return {"intent": "unknown", "confidence": 0.0, "all_probabilities": {}, "model_type": "error"}
```

---

## **Key Learning Points**

### **1. Data Pipeline**
- Raw text → Cleaning → Tokenization → Lemmatization → Sequences
- Always preprocess prediction inputs the same way as training data

### **2. Model Selection**
- **CNN**: Good for local patterns (phrases)
- **LSTM**: Good for sequence context  
- **Transformer**: Good for complex relationships
- **Ensemble**: Combines strengths of all

### **3. Training Best Practices**
- Use validation set for tuning
- Implement early stopping
- Use learning rate scheduling
- Handle class imbalance

### **4. Evaluation**
- Always test on unseen data
- Use multiple metrics (accuracy, precision, recall)
- Compare single vs ensemble performance

This `jobs.py` script provides a complete, production-ready training pipeline for an advanced chatbot system!