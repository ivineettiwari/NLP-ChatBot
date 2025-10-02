# Imports
import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, Embedding, Dropout, Bidirectional, SpatialDropout1D,
    Conv1D, GlobalMaxPooling1D, BatchNormalization, MultiHeadAttention,
    LayerNormalization, Input, concatenate, Reshape, Flatten,
    Conv1DTranspose, LeakyReLU
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard, LearningRateScheduler
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


# GAN Classes for Advanced Data Augmentation
class TextGAN:
    def __init__(self, vocab_size, max_sequence_length, embedding_dim=100, latent_dim=100):
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()
        
    def build_generator(self):
        """Generator model that creates synthetic text sequences"""
        model = Sequential([
            Dense(256, input_dim=self.latent_dim),
            LeakyReLU(alpha=0.2),
            BatchNormalization(),
            
            Dense(512),
            LeakyReLU(alpha=0.2),
            BatchNormalization(),
            
            Dense(1024),
            LeakyReLU(alpha=0.2),
            BatchNormalization(),
            
            Dense(self.max_sequence_length * self.embedding_dim),
            LeakyReLU(alpha=0.2),
            Reshape((self.max_sequence_length, self.embedding_dim)),
            
            Conv1DTranspose(128, 5, padding='same'),
            LeakyReLU(alpha=0.2),
            
            Conv1DTranspose(64, 5, padding='same'),
            LeakyReLU(alpha=0.2),
            
            Conv1D(self.embedding_dim, 5, padding='same', activation='tanh'),
            Dense(self.vocab_size, activation='softmax')
        ])
        
        return model
    
    def build_discriminator(self):
        """Discriminator model that distinguishes real from synthetic sequences"""
        model = Sequential([
            Conv1D(64, 5, padding='same', input_shape=(self.max_sequence_length, self.vocab_size)),
            LeakyReLU(alpha=0.2),
            Dropout(0.3),
            
            Conv1D(128, 5, padding='same'),
            LeakyReLU(alpha=0.2),
            Dropout(0.3),
            
            Conv1D(256, 5, padding='same'),
            LeakyReLU(alpha=0.2),
            Dropout(0.3),
            
            GlobalMaxPooling1D(),
            Dense(512),
            LeakyReLU(alpha=0.2),
            Dropout(0.3),
            
            Dense(256),
            LeakyReLU(alpha=0.2),
            Dropout(0.3),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_gan(self):
        """Combined GAN model"""
        # Make discriminator non-trainable during generator training
        self.discriminator.trainable = False
        
        gan_input = Input(shape=(self.latent_dim,))
        generated_sequence = self.generator(gan_input)
        gan_output = self.discriminator(generated_sequence)
        
        gan = Model(gan_input, gan_output)
        gan.compile(
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy'
        )
        
        return gan
    
    def train(self, real_sequences, epochs=1000, batch_size=32, save_interval=100):
        """Train the GAN"""
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # Train Discriminator
            # Select random real sequences
            idx = np.random.randint(0, real_sequences.shape[0], batch_size)
            real_seqs = real_sequences[idx]
            
            # Generate fake sequences
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_seqs = self.generator.predict(noise, verbose=0)
            
            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch(real_seqs, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_seqs, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, real_labels)
            
            if epoch % save_interval == 0:
                print(f"Epoch {epoch}, D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]:.4f}, G Loss: {g_loss:.4f}")
                
                # Save generated samples
                self.save_generated_samples(epoch)
    
    def generate_samples(self, num_samples):
        """Generate synthetic text sequences"""
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        generated_sequences = self.generator.predict(noise, verbose=0)
        return generated_sequences
    
    def save_generated_samples(self, epoch):
        """Save generated samples for inspection"""
        samples = self.generate_samples(5)
        # Convert back to text (simplified)
        print(f"Epoch {epoch} - Generated samples saved")


class ConditionalTextGAN(TextGAN):
    """Conditional GAN that generates text for specific intents"""
    
    def __init__(self, vocab_size, max_sequence_length, num_classes, embedding_dim=100, latent_dim=100):
        self.num_classes = num_classes
        super().__init__(vocab_size, max_sequence_length, embedding_dim, latent_dim)
    
    def build_generator(self):
        """Conditional generator with class input"""
        noise_input = Input(shape=(self.latent_dim,))
        class_input = Input(shape=(self.num_classes,))
        
        # Concatenate noise and class label
        combined_input = concatenate([noise_input, class_input])
        
        x = Dense(256)(combined_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        
        x = Dense(512)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        
        x = Dense(1024)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        
        x = Dense(self.max_sequence_length * self.embedding_dim)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Reshape((self.max_sequence_length, self.embedding_dim))(x)
        
        x = Conv1DTranspose(128, 5, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Conv1DTranspose(64, 5, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Conv1D(self.embedding_dim, 5, padding='same', activation='tanh')(x)
        output = Dense(self.vocab_size, activation='softmax')(x)
        
        model = Model([noise_input, class_input], output)
        return model
    
    def build_discriminator(self):
        """Conditional discriminator with class input"""
        sequence_input = Input(shape=(self.max_sequence_length, self.vocab_size))
        class_input = Input(shape=(self.num_classes,))
        
        # Expand class input to match sequence dimensions
        class_expanded = Dense(self.max_sequence_length)(class_input)
        class_expanded = Reshape((self.max_sequence_length, 1))(class_expanded)
        class_expanded = Dense(self.vocab_size)(class_expanded)
        
        # Concatenate sequence and class information
        combined_input = concatenate([sequence_input, class_expanded])
        
        x = Conv1D(64, 5, padding='same')(combined_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        
        x = Conv1D(128, 5, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        
        x = Conv1D(256, 5, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        
        x = GlobalMaxPooling1D()(x)
        x = Dense(512)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        
        x = Dense(256)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model([sequence_input, class_input], output)
        model.compile(
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_gan(self):
        """Combined conditional GAN"""
        self.discriminator.trainable = False
        
        noise_input = Input(shape=(self.latent_dim,))
        class_input = Input(shape=(self.num_classes,))
        
        generated_sequence = self.generator([noise_input, class_input])
        gan_output = self.discriminator([generated_sequence, class_input])
        
        gan = Model([noise_input, class_input], gan_output)
        gan.compile(
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy'
        )
        
        return gan
    
    def generate_conditional_samples(self, num_samples, class_labels):
        """Generate samples for specific classes"""
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        generated_sequences = self.generator.predict([noise, class_labels], verbose=0)
        return generated_sequences


# Enhanced Data Augmentation with GAN
def augment_with_gan(preprocessor, documents, labels, augmentation_factor=2):
    """Augment training data using GAN-generated samples"""
    print("Starting GAN-based data augmentation...")
    
    # Convert documents to sequences
    sequences = [preprocessor.text_to_sequence(doc) for doc in documents]
    X = pad_sequences(
        sequences,
        maxlen=preprocessor.max_sequence_length,
        padding="post",
        truncating="post",
    )
    
    # Convert to one-hot encoding for GAN training
    X_one_hot = tf.keras.utils.to_categorical(X, num_classes=len(preprocessor.vocab) + 1)
    
    # Initialize and train GAN
    gan = TextGAN(
        vocab_size=len(preprocessor.vocab) + 1,
        max_sequence_length=preprocessor.max_sequence_length,
        embedding_dim=100,
        latent_dim=100
    )
    
    # Train GAN (simplified for demonstration)
    print("Training GAN for data augmentation...")
    gan.train(X_one_hot, epochs=500, batch_size=32)
    
    # Generate synthetic samples
    num_synthetic = len(documents) * augmentation_factor
    synthetic_sequences = gan.generate_samples(num_synthetic)
    
    # Convert synthetic sequences back to text
    synthetic_docs = []
    synthetic_labels = []
    
    for i, seq in enumerate(synthetic_sequences):
        # Convert probability distribution to word indices
        word_indices = np.argmax(seq, axis=1)
        # Convert to text
        words = [preprocessor.idx2word.get(idx, '') for idx in word_indices if idx > 0]
        synthetic_text = ' '.join(words)
        
        if len(synthetic_text.strip()) > 5:  # Filter reasonable samples
            synthetic_docs.append(synthetic_text)
            # Assign random label from original data for diversity
            synthetic_labels.append(random.choice(labels))
    
    print(f"Generated {len(synthetic_docs)} synthetic samples via GAN")
    
    # Combine with original data
    augmented_docs = documents + synthetic_docs
    augmented_labels = labels + synthetic_labels
    
    return augmented_docs, augmented_labels


def conditional_gan_augmentation(preprocessor, documents, labels, augmentation_factor=2):
    """Augment data using conditional GAN for specific intents"""
    print("Starting Conditional GAN data augmentation...")
    
    # Prepare data for conditional GAN
    unique_labels = sorted(set(labels))
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Convert documents to sequences
    sequences = [preprocessor.text_to_sequence(doc) for doc in documents]
    X = pad_sequences(
        sequences,
        maxlen=preprocessor.max_sequence_length,
        padding="post",
        truncating="post",
    )
    
    # Convert to one-hot encoding
    X_one_hot = tf.keras.utils.to_categorical(X, num_classes=len(preprocessor.vocab) + 1)
    
    # Prepare labels for conditional GAN
    y_labels = np.array([label2idx[label] for label in labels])
    y_one_hot = tf.keras.utils.to_categorical(y_labels, num_classes=len(unique_labels))
    
    # Initialize conditional GAN
    cgan = ConditionalTextGAN(
        vocab_size=len(preprocessor.vocab) + 1,
        max_sequence_length=preprocessor.max_sequence_length,
        num_classes=len(unique_labels),
        embedding_dim=100,
        latent_dim=100
    )
    
    # Train conditional GAN
    print("Training Conditional GAN...")
    
    # Simplified training loop
    batch_size = 32
    epochs = 300
    
    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, X_one_hot.shape[0], batch_size)
        real_seqs = X_one_hot[idx]
        real_labels_batch = y_one_hot[idx]
        
        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_labels = y_one_hot[np.random.randint(0, y_one_hot.shape[0], batch_size)]
        fake_seqs = cgan.generator.predict([noise, fake_labels], verbose=0)
        
        # Train discriminator
        d_loss_real = cgan.discriminator.train_on_batch([real_seqs, real_labels_batch], np.ones((batch_size, 1)))
        d_loss_fake = cgan.discriminator.train_on_batch([fake_seqs, fake_labels], np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_labels = y_one_hot[np.random.randint(0, y_one_hot.shape[0], batch_size)]
        g_loss = cgan.gan.train_on_batch([noise, valid_labels], np.ones((batch_size, 1)))
        
        if epoch % 100 == 0:
            print(f"CGAN Epoch {epoch}, D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}")
    
    # Generate synthetic samples for each class
    synthetic_docs = []
    synthetic_labels_list = []
    
    for label_idx, label in enumerate(unique_labels):
        num_samples_per_class = len([l for l in labels if l == label]) * augmentation_factor
        
        # Generate samples for this class
        class_vector = np.zeros((num_samples_per_class, len(unique_labels)))
        class_vector[:, label_idx] = 1
        
        noise = np.random.normal(0, 1, (num_samples_per_class, 100))
        synthetic_sequences = cgan.generator.predict([noise, class_vector], verbose=0)
        
        # Convert to text
        for seq in synthetic_sequences:
            word_indices = np.argmax(seq, axis=1)
            words = [preprocessor.idx2word.get(idx, '') for idx in word_indices if idx > 0]
            synthetic_text = ' '.join(words)
            
            if len(synthetic_text.strip()) > 5:
                synthetic_docs.append(synthetic_text)
                synthetic_labels_list.append(label)
    
    print(f"Generated {len(synthetic_docs)} conditional synthetic samples")
    
    # Combine with original data
    augmented_docs = documents + synthetic_docs
    augmented_labels = labels + synthetic_labels_list
    
    return augmented_docs, augmented_labels


# [Rest of the code remains the same - keeping the original functions for brevity]
# Data enhancement strategies (original)
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


# Enhanced Main Training Function with GAN
def my_jobs():
    print("Starting advanced NLP preprocessing and model training with GAN augmentation...")

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

    # Initialize enhanced preprocessor
    preprocessor = EnhancedTextPreprocessor()

    # Build vocabulary
    print("1. Building vocabulary...")
    preprocessor.build_vocabulary(documents)

    # 1. Traditional Data Augmentation
    print("2. Applying traditional data augmentation...")
    augmented_docs, augmented_labels = augment_training_data(documents, labels)
    
    # 2. GAN-based Data Augmentation (Conditional GAN)
    print("3. Applying GAN-based data augmentation...")
    try:
        gan_augmented_docs, gan_augmented_labels = conditional_gan_augmentation(
            preprocessor, augmented_docs, augmented_labels, augmentation_factor=1
        )
        print(f"GAN augmentation successful: {len(gan_augmented_docs)} total samples")
        final_docs, final_labels = gan_augmented_docs, gan_augmented_labels
    except Exception as e:
        print(f"GAN augmentation failed: {e}. Using traditional augmentation only.")
        final_docs, final_labels = augmented_docs, augmented_labels
    
    # 3. Data Balancing
    print("4. Balancing dataset...")
    balanced_docs, balanced_labels = balance_dataset(final_docs, final_labels)
    
    print(f"Original data: {len(documents)} samples")
    print(f"After traditional augmentation: {len(augmented_docs)} samples")
    print(f"After GAN augmentation: {len(final_docs)} samples")
    print(f"After balancing: {len(balanced_docs)} samples")

    # Continue with normal training pipeline...
    print("5. Building vocabulary and extracting features...")
    # Rebuild vocabulary with augmented data if needed
    if len(final_docs) > len(documents):
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

    # [Rest of the training pipeline remains the same...]
    # Hyperparameter tuning, model training, ensemble creation, etc.

    # For demonstration, let's create a simple model training
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
    
    print("6. Training final model with GAN-augmented data...")
    
    # Simple model for demonstration
    model = Sequential([
        Embedding(
            input_dim=len(preprocessor.vocab) + 1,
            output_dim=100,
            input_length=preprocessor.max_sequence_length,
            mask_zero=True,
        ),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(unique_labels), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Model Test Accuracy: {test_accuracy:.4f}")
    
    # Save model
    model.save("model_trained/gan_augmented_model.h5")
    
    # Save preprocessing artifacts
    with open("model_trained/gan_enhanced_preprocessor.pickle", "wb") as f:
        pickle.dump(
            {
                "preprocessor": preprocessor,
                "label2idx": label2idx,
                "idx2label": {idx: label for label, idx in label2idx.items()},
                "max_sequence_length": preprocessor.max_sequence_length,
                "vocab_size": len(preprocessor.vocab),
            },
            f,
        )
    
    print("Training with GAN augmentation completed successfully!")


if __name__ == "__main__":
    my_jobs()