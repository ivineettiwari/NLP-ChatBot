import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, Embedding, Dropout, Bidirectional, 
    Conv1D, GlobalMaxPooling1D, BatchNormalization, MultiHeadAttention,
    LayerNormalization, Input, concatenate, Reshape, Flatten,
    Conv1DTranspose, LeakyReLU
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard
)
import nltk
import json
import pickle
import re
import random
from collections import Counter
import matplotlib.pyplot as plt

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

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

    def clean_text(self, text):
        """Enhanced text cleaning"""
        if not isinstance(text, str):
            return ""
            
        text = text.lower()
        # Keep basic punctuation for better context
        text = re.sub(r"[^a-zA-Z\s\?\!\.\,]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def advanced_tokenize(self, text):
        """Tokenization with POS tagging and lemmatization"""
        if not text:
            return []
            
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 1]

        try:
            pos_tags = nltk.pos_tag(tokens)
            lemmatized_tokens = []
            
            for token, pos in pos_tags:
                pos_tag = wordnet.VERB
                if pos.startswith('J'):  # Adjective
                    pos_tag = wordnet.ADJ
                elif pos.startswith('V'):  # Verb
                    pos_tag = wordnet.VERB
                elif pos.startswith('R'):  # Adverb
                    pos_tag = wordnet.ADV
                else:  # Default to noun
                    pos_tag = wordnet.NOUN
                    
                lemma = self.lemmatizer.lemmatize(token, pos=pos_tag)
                lemmatized_tokens.append(lemma)
                
            return lemmatized_tokens
        except Exception:
            return [self.lemmatizer.lemmatize(token) for token in tokens]

    def build_vocabulary(self, documents):
        """Build vocabulary from documents"""
        all_tokens = []
        for doc in documents:
            cleaned_text = self.clean_text(doc)
            tokens = self.advanced_tokenize(cleaned_text)
            all_tokens.extend(tokens)
            self.vocab.update(tokens)

        # Create word mappings
        self.vocab = sorted(self.vocab)
        self.word2idx = {word: idx + 1 for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        # Calculate optimal sequence length
        sequence_lengths = [len(self.advanced_tokenize(self.clean_text(doc))) for doc in documents]
        self.max_sequence_length = max(10, int(np.percentile(sequence_lengths, 90))) if sequence_lengths else 50

        return all_tokens

    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        cleaned_text = self.clean_text(text)
        tokens = self.advanced_tokenize(cleaned_text)
        return [self.word2idx.get(token, 0) for token in tokens]

class SimplifiedTextGAN:
    """Simplified GAN for practical text generation"""
    def __init__(self, vocab_size, max_sequence_length, embedding_dim=50, latent_dim=50):
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        
    def build_generator(self):
        """Simplified generator"""
        model = Sequential([
            Dense(128, input_dim=self.latent_dim),
            LeakyReLU(0.2),
            BatchNormalization(),
            
            Dense(256),
            LeakyReLU(0.2),
            BatchNormalization(),
            
            Dense(self.max_sequence_length * self.embedding_dim),
            LeakyReLU(0.2),
            Reshape((self.max_sequence_length, self.embedding_dim)),
            
            Dense(self.vocab_size, activation='softmax')
        ])
        return model
    
    def build_discriminator(self):
        """Simplified discriminator"""
        model = Sequential([
            Dense(256, input_shape=(self.max_sequence_length, self.vocab_size)),
            LeakyReLU(0.2),
            Flatten(),
            
            Dense(128),
            LeakyReLU(0.2),
            Dropout(0.3),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(0.0002, 0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

def advanced_data_augmentation(documents, labels, augmentation_factor=3):
    """Enhanced data augmentation without GAN complexity"""
    augmented_docs = documents.copy()
    augmented_labels = labels.copy()
    
    print("Applying advanced data augmentation...")
    
    for doc, label in zip(documents, labels):
        for _ in range(augmentation_factor):
            # Multiple augmentation techniques
            try:
                # Synonym replacement
                aug_doc = synonym_replacement(doc)
                if aug_doc != doc and len(aug_doc) > 10:
                    augmented_docs.append(aug_doc)
                    augmented_labels.append(label)
            except:
                pass
                
            try:
                # Random swap
                aug_doc = random_swap(doc)
                if aug_doc != doc:
                    augmented_docs.append(aug_doc)
                    augmented_labels.append(label)
            except:
                pass
                
            try:
                # Random deletion
                aug_doc = random_deletion(doc)
                if aug_doc != doc and len(aug_doc) > 10:
                    augmented_docs.append(aug_doc)
                    augmented_labels.append(label)
            except:
                pass
    
    print(f"Augmented from {len(documents)} to {len(augmented_docs)} samples")
    return augmented_docs, augmented_labels

def synonym_replacement(text, n=2):
    """Replace n words with synonyms"""
    words = text.split()
    if len(words) < 2:
        return text
        
    new_words = words.copy()
    random_word_list = [word for word in words if word not in stopwords.words('english')]
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if synonyms:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
            
    return ' '.join(new_words)

def get_synonyms(word):
    """Get synonyms for a word"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            if synonym != word and len(synonym.split()) == 1:
                synonyms.add(synonym)
    return list(synonyms)

def random_swap(text, n=1):
    """Randomly swap two words n times"""
    words = text.split()
    if len(words) < 2:
        return text
        
    new_words = words.copy()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return ' '.join(new_words)

def random_deletion(text, p=0.2):
    """Randomly delete words with probability p"""
    words = text.split()
    if len(words) < 2:
        return text
        
    new_words = []
    for word in words:
        if random.random() > p:
            new_words.append(word)
            
    # If everything deleted, return original
    if len(new_words) == 0:
        return ' '.join(words)
    return ' '.join(new_words)

def create_advanced_model(vocab_size, sequence_length, num_classes):
    """Create an advanced neural network model"""
    
    # Text input
    text_input = Input(shape=(sequence_length,))
    
    # Embedding layer
    embedding = Embedding(
        input_dim=vocab_size + 1,
        output_dim=128,
        input_length=sequence_length,
        mask_zero=True,
        name="embedding"
    )(text_input)
    
    # Multiple parallel processing paths
    # 1. BiLSTM path
    lstm_path = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(embedding)
    lstm_path = Bidirectional(LSTM(32, dropout=0.2))(lstm_path)
    
    # 2. CNN path
    conv_path = Conv1D(64, 3, activation='relu', padding='same')(embedding)
    conv_path = GlobalMaxPooling1D()(conv_path)
    
    # 3. Attention path
    attention_path = MultiHeadAttention(num_heads=4, key_dim=32)(embedding, embedding)
    attention_path = GlobalMaxPooling1D()(attention_path)
    
    # Combine all paths
    combined = concatenate([lstm_path, conv_path, attention_path])
    
    # Dense layers
    x = Dense(128, activation='relu')(combined)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=text_input, outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_advanced_model():
    """Main training function"""
    print("Starting advanced model training...")
    
    # Load data
    try:
        with open("data/final.json", "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Error: final.json not found. Creating sample data for demonstration.")
        # Create sample data for demonstration
        data = {
            "intents": [
                {
                    "tag": "greeting",
                    "patterns": ["hello", "hi", "hey", "good morning", "good afternoon"]
                },
                {
                    "tag": "goodbye", 
                    "patterns": ["bye", "goodbye", "see you", "take care"]
                },
                {
                    "tag": "thanks",
                    "patterns": ["thank you", "thanks", "appreciate it", "thank you very much"]
                }
            ]
        }
    
    # Extract data
    documents = []
    labels = []
    
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            documents.append(pattern)
            labels.append(intent["tag"])
    
    print(f"Loaded {len(documents)} training samples")
    print(f"Classes: {set(labels)}")
    
    # Data augmentation
    augmented_docs, augmented_labels = advanced_data_augmentation(documents, labels)
    
    # Initialize and fit preprocessor
    preprocessor = EnhancedTextPreprocessor()
    preprocessor.build_vocabulary(augmented_docs)
    
    # Prepare sequences
    sequences = [preprocessor.text_to_sequence(doc) for doc in augmented_docs]
    X = pad_sequences(
        sequences,
        maxlen=preprocessor.max_sequence_length,
        padding="post",
        truncating="post"
    )
    
    # Prepare labels
    unique_labels = sorted(set(augmented_labels))
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label2idx[label] for label in augmented_labels])
    y_categorical = tf.keras.utils.to_categorical(y, num_classes=len(unique_labels))
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_categorical, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Vocabulary size: {len(preprocessor.vocab)}")
    
    # Create and train model
    model = create_advanced_model(
        vocab_size=len(preprocessor.vocab),
        sequence_length=preprocessor.max_sequence_length,
        num_classes=len(unique_labels)
    )
    
    print("Model architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5),
        ModelCheckpoint(
            "model_trained/best_model.h5",
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
    
    # Save model and artifacts
    model.save("model_trained/advanced_chatbot_model.h5")
    
    artifacts = {
        "preprocessor": preprocessor,
        "label2idx": label2idx,
        "idx2label": {idx: label for label, idx in label2idx.items()},
        "max_sequence_length": preprocessor.max_sequence_length,
        "vocab_size": len(preprocessor.vocab),
    }
    
    with open("model_trained/chatbot_artifacts.pickle", "wb") as f:
        pickle.dump(artifacts, f)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_trained/training_history.png')
    plt.show()
    
    print("Training completed successfully!")
    return model, artifacts

class ChatbotPredictor:
    """Class for making predictions with the trained model"""
    
    def __init__(self, model_path="model_trained/advanced_chatbot_model.h5", 
                 artifacts_path="model_trained/chatbot_artifacts.pickle"):
        self.model = tf.keras.models.load_model(model_path)
        with open(artifacts_path, "rb") as f:
            artifacts = pickle.load(f)
            self.preprocessor = artifacts["preprocessor"]
            self.idx2label = artifacts["idx2label"]
            self.max_sequence_length = artifacts["max_sequence_length"]
    
    def predict_intent(self, text):
        """Predict intent for given text"""
        sequence = self.preprocessor.text_to_sequence(text)
        padded_sequence = pad_sequences(
            [sequence],
            maxlen=self.max_sequence_length,
            padding="post",
            truncating="post"
        )
        
        prediction = self.model.predict(padded_sequence, verbose=0)
        predicted_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_idx]
        
        return self.idx2label[predicted_idx], confidence

# Example usage
if __name__ == "__main__":
    # Train the model
    model, artifacts = train_advanced_model()
    
    # Test the predictor
    predictor = ChatbotPredictor()
    
    test_phrases = [
        "hello there",
        "thanks for your help", 
        "goodbye for now",
        "what's the weather like?"
    ]
    
    print("\nTesting the chatbot:")
    for phrase in test_phrases:
        intent, confidence = predictor.predict_intent(phrase)
        print(f"'{phrase}' -> {intent} (confidence: {confidence:.3f})")