# Imports
import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Embedding,
    Dropout,
    Bidirectional,
    SpatialDropout1D,
    Conv1D,
    GlobalMaxPooling1D,
    BatchNormalization,
    MultiHeadAttention,
    LayerNormalization,
    Input,
    concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
    LearningRateScheduler,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize


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

    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        cleaned_text = self.clean_text(text)
        tokens = self.advanced_tokenize(cleaned_text)
        sequence = [self.word2idx.get(token, 0) for token in tokens]  # 0 for OOV
        return sequence


# 1. DATA ENHANCEMENT STRATEGIES
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

def random_insertion(text: str, n: int = 1) -> str:
    """Randomly insert synonyms"""
    words = text.split()
    new_words = words.copy()
    
    for _ in range(n):
        if not words:
            break
        random_word = random.choice(words)
        synonyms = get_synonyms(random_word)
        if synonyms:
            synonym = random.choice(synonyms)
            random_idx = random.randint(0, len(new_words))
            new_words.insert(random_idx, synonym)
    
    return ' '.join(new_words)

def paraphrase_text(text: str) -> str:
    """Simple paraphrasing by reordering and synonym replacement"""
    words = text.split()
    if len(words) <= 1:
        return text
    
    # Randomly reorder parts of the sentence
    if len(words) > 3:
        split_point = random.randint(1, len(words)-2)
        new_words = words[split_point:] + words[:split_point]
        return ' '.join(new_words)
    else:
        return text

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


# 2. ADVANCED MODEL ARCHITECTURES
def create_advanced_model(vocab_size: int, num_classes: int, max_sequence_length: int, 
                         embedding_dim: int = 300, dropout_rate: float = 0.3, learning_rate: float = 0.001):
    """Create a more sophisticated model architecture"""
    
    model = Sequential([
        # Enhanced embedding layer
        Embedding(
            input_dim=vocab_size + 1,
            output_dim=embedding_dim,
            input_length=max_sequence_length,
            mask_zero=True,
        ),
        
        SpatialDropout1D(dropout_rate),
        
        # Multi-scale feature extraction
        Conv1D(64, 2, activation='relu', padding='same'),
        Conv1D(64, 3, activation='relu', padding='same'),
        Conv1D(64, 4, activation='relu', padding='same'),
        
        # Bidirectional LSTM
        Bidirectional(LSTM(
            128, 
            return_sequences=True, 
            dropout=0.2, 
            recurrent_dropout=0.2,
            kernel_regularizer=l2(0.01)
        )),
        
        # Global pooling
        GlobalMaxPooling1D(),
        
        # Dense layers with batch normalization
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def create_transformer_model(vocab_size: int, num_classes: int, max_sequence_length: int, embedding_dim: int = 128):
    """Transformer-based model for better context understanding"""
    
    inputs = Input(shape=(max_sequence_length,))
    
    # Embedding layer
    embedding = Embedding(vocab_size + 1, embedding_dim)(inputs)
    embedding = SpatialDropout1D(0.2)(embedding)
    
    # Transformer block
    attention_output = MultiHeadAttention(
        num_heads=8, 
        key_dim=embedding_dim // 8
    )(embedding, embedding)
    
    attention_output = Dropout(0.1)(attention_output)
    attention_output = LayerNormalization()(embedding + attention_output)
    
    # Feed forward
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

def create_cnn_model(vocab_size: int, num_classes: int, max_sequence_length: int, embedding_dim: int = 100):
    """CNN-based model for text classification"""
    
    model = Sequential([
        Embedding(
            input_dim=vocab_size + 1,
            output_dim=embedding_dim,
            input_length=max_sequence_length,
        ),
        
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# 3. ADVANCED TRAINING TECHNIQUES
def train_advanced_model(model, X_train, y_train, X_val, y_val, epochs: int = 200):
    """Enhanced training with advanced techniques"""
    
    def lr_scheduler(epoch, lr):
        if epoch < 10:
            return float(lr)  # Convert to float
        else:
            return float(lr * tf.math.exp(-0.1))  # Convert to float
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            min_delta=0.001,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'model_trained/best_model_advanced.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        LearningRateScheduler(lr_scheduler)
    ]
    
    # Class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(np.argmax(y_train, axis=1)),
        y=np.argmax(y_train, axis=1)
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
        shuffle=True
    )
    
    return history


# 4. ENSEMBLE METHODS
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
                verbose=0,
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
        
        # Weighted average (you can adjust weights based on model performance)
        weights = [0.4, 0.4, 0.2]  # Adjust based on validation performance
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        return weighted_pred
    
    def save_ensemble(self, directory: str):
        """Save all ensemble models"""
        os.makedirs(directory, exist_ok=True)
        for name, model in self.models.items():
            model.save(f"{directory}/{name}.h5")
            
    def load_ensemble(self, directory: str):
        """Load ensemble models"""
        for name in self.models.keys():
            self.models[name] = tf.keras.models.load_model(f"{directory}/{name}.h5")


# 5. SIMPLIFIED HYPERPARAMETER OPTIMIZATION
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
        
        # Train for fewer epochs for quick evaluation
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
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


# 6. COMPREHENSIVE EVALUATION
def comprehensive_evaluation(model, X_test, y_test, label_names: List[str]):
    """Comprehensive model evaluation"""
    
    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Multiple metrics
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    report = classification_report(y_true_classes, y_pred_classes, target_names=label_names)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('model_trained/confusion_matrix.png')
    plt.close()
    
    return accuracy, report, cm


# MAIN TRAINING FUNCTION WITH ALL IMPROVEMENTS
def my_jobs():
    print("Starting advanced NLP preprocessing and model training with all improvements...")

    # Load data
    with open("data/final.json", encoding="utf-8") as file:
        data = json.load(file)

    # Extract patterns and labels
    documents = []
    labels = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            documents.append(pattern)
            labels.append(intent["tag"])

    # 1. DATA AUGMENTATION
    print("1. Applying data augmentation...")
    augmented_docs, augmented_labels = augment_training_data(documents, labels)
    
    # 2. DATA BALANCING
    print("2. Balancing dataset...")
    balanced_docs, balanced_labels = balance_dataset(augmented_docs, augmented_labels)
    
    print(f"Original data: {len(documents)} samples")
    print(f"After augmentation: {len(augmented_docs)} samples")
    print(f"After balancing: {len(balanced_docs)} samples")

    # Initialize enhanced preprocessor
    preprocessor = EnhancedTextPreprocessor()

    # Build vocabulary
    print("3. Building vocabulary and extracting features...")
    preprocessor.build_vocabulary(balanced_docs)

    # Convert documents to sequences
    sequences = [preprocessor.text_to_sequence(doc) for doc in balanced_docs]

    # Pad sequences
    X = pad_sequences(
        sequences,
        maxlen=preprocessor.max_sequence_length,
        padding="post",
        truncating="post",
    )

    # Prepare labels
    unique_labels = sorted(set(balanced_labels))
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label2idx[label] for label in balanced_labels])

    # Convert to categorical
    y_categorical = tf.keras.utils.to_categorical(y, num_classes=len(unique_labels))

    # Split data
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

    # 3. MANUAL HYPERPARAMETER TUNING (Simplified)
    print("4. Starting manual hyperparameter tuning...")
    best_config, tuned_model = manual_hyperparameter_tuning(
        X_train, y_train, X_val, y_val, 
        len(preprocessor.vocab), len(unique_labels), preprocessor.max_sequence_length
    )

    # 4. CREATE AND TRAIN ADVANCED MODEL WITH BEST CONFIG
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

    # Train with advanced techniques
    print("Training advanced model...")
    history = train_advanced_model(advanced_model, X_train, y_train, X_val, y_val, epochs=150)

    # 5. CREATE AND TRAIN ENSEMBLE
    print("6. Creating model ensemble...")
    ensemble = ModelEnsemble(
        vocab_size=len(preprocessor.vocab),
        num_classes=len(unique_labels),
        max_sequence_length=preprocessor.max_sequence_length
    )
    ensemble.create_ensemble()
    ensemble.train_ensemble(X_train, y_train, X_val, y_val, epochs=80)

    # 6. COMPREHENSIVE EVALUATION
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

    # Save everything
    print("8. Saving models and artifacts...")
    
    # Save single model
    advanced_model.save("model_trained/advanced_model_improved.h5")
    
    # Save ensemble
    ensemble.save_ensemble("model_trained/ensemble_models")
    
    # Save preprocessing artifacts
    with open("model_trained/enhanced_preprocessor.pickle", "wb") as f:
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
        plt.savefig("model_trained/improved_training_history.png")
        plt.close()

    print("\n=== TRAINING SUMMARY ===")
    print(f"Single Model Accuracy: {single_accuracy:.4f}")
    print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
    print(f"Improvement: {((ensemble_accuracy - single_accuracy) / single_accuracy * 100):.2f}%")
    print(f"Best Configuration: {best_config['description']}")
    print("Training completed successfully!")


def predict_intent_improved(
    text,
    model_path="model_trained/advanced_model_improved.h5",
    preprocessor_path="model_trained/enhanced_preprocessor.pickle",
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
            ensemble.load_ensemble("model_trained/ensemble_models")
            
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


if __name__ == "__main__":
    my_jobs()

    # Test predictions
    test_texts = [
        "Hello, how are you doing today?",
        "I want to book a table for dinner",
        "What's on your menu?",
        "What are your operating hours?"
    ]
    
    print("\n=== TEST PREDICTIONS ===")
    for test_text in test_texts:
        # Single model prediction
        result_single = predict_intent_improved(test_text, use_ensemble=False)
        # Ensemble prediction
        result_ensemble = predict_intent_improved(test_text, use_ensemble=True)
        
        print(f"\nText: '{test_text}'")
        print(f"Single Model: {result_single['intent']} (confidence: {result_single['confidence']:.4f})")
        print(f"Ensemble: {result_ensemble['intent']} (confidence: {result_ensemble['confidence']:.4f})")