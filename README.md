# Restaurant Chatbot - Advanced NLP with TensorFlow & FastAPI

An intelligent restaurant chatbot built with advanced Natural Language Processing (NLP) techniques, deep learning models, and modern web technologies. This chatbot can handle restaurant inquiries, table reservations, menu information, and provide engaging customer service.

## 🚀 Features

- **Advanced NLP Processing**: Tokenization, lemmatization, POS tagging, and text cleaning
- **Multiple Model Architectures**: CNN, LSTM, Transformer, and Ensemble models
- **Real-time Streaming**: Server-Sent Events for live response streaming
- **Modern Web Framework**: FastAPI with async/await support
- **Data Augmentation**: Automatic training data enhancement
- **Hyperparameter Optimization**: Automated model tuning
- **Comprehensive Evaluation**: Multiple metrics and visualization

## 🏗️ Architecture

### Model Stack
- **EnhancedTextPreprocessor**: Advanced NLP preprocessing with POS tagging
- **CNN + LSTM Hybrid**: Multi-scale feature extraction with bidirectional context
- **Transformer Model**: Self-attention mechanism for better context understanding
- **Model Ensemble**: Weighted combination of multiple architectures
- **FastAPI Backend**: High-performance async web server

### Key Technologies
- **TensorFlow/Keras**: Deep learning framework
- **NLTK**: Natural Language Processing
- **FastAPI**: Modern web framework
- **Scikit-learn**: Machine learning utilities
- **NumPy/Pandas**: Data processing

## 📁 Project Structure

```
chatbot-project/
├── filesTech/
│   ├── final.json                    # Training data with intents
│   ├── enhanced_preprocessor.pickle  # Saved preprocessor
│   ├── advanced_model_improved.h5    # Single trained model
│   └── ensemble_models/              # Ensemble model directory
│       ├── advanced_model.h5
│       ├── transformer_model.h5
│       └── cnn_model.h5
├── static/
│   ├── css/
│   │   └── style.css                 # Chat interface styling
│   └── js/
│       └── script.js                 # Frontend JavaScript
├── templates/
│   └── index.html                    # Chat interface HTML
├── jobs.py                          # Model training script
├── app.py                           # FastAPI server
└── requirements.txt                 # Python dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.12
- pip package manager

### Step 1: Clone and Setup
```bash
# Clone the repository
git clone https://github.com/ivineettiwari/NLP-ChatBot.git
cd .\NLP-ChatBot\

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### Step 3: Prepare Training Data
Create `filesTech/final.json` with your training data:
```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey", "Good morning"],
      "responses": ["Hello! Welcome to our restaurant!"]
    },
    {
      "tag": "book_table",
      "patterns": ["Book a table", "I want to reserve"],
      "responses": ["I'd be happy to help you book a table!"]
    }
  ]
}
```

## 🚀 Quick Start

### Option 1: Train New Models
```bash
# Train the chatbot models
python jobs.py
```

This will:
- Preprocess and augment training data
- Perform hyperparameter optimization
- Train single and ensemble models
- Save models and preprocessor
- Generate evaluation reports

### Option 2: Run Chatbot Server
```bash
# Start the FastAPI server
python app.py
```

Server will start at: `http://localhost:8000`

## 📊 Training Process

### Data Processing Pipeline
1. **Text Cleaning**: Lowercase, remove special characters, normalize whitespace
2. **Tokenization**: Split text into words with POS tagging
3. **Lemmatization**: Convert words to base form using WordNet
4. **Vocabulary Building**: Create word-to-index mapping
5. **Sequence Padding**: Standardize input lengths

### Model Training Features
- **Data Augmentation**: Synonym replacement, random insertion, paraphrasing
- **Class Balancing**: Automatic handling of imbalanced datasets
- **Hyperparameter Tuning**: Manual optimization of embedding dimensions, dropout rates, learning rates
- **Ensemble Learning**: Combined predictions from multiple model architectures
- **Advanced Callbacks**: Early stopping, learning rate scheduling, model checkpointing

## 🌐 API Endpoints

### Chat Endpoints
- `POST /chat` - Get chatbot response (JSON)
- `GET /stream?message=...` - Stream response with Server-Sent Events
- `GET /` - Web chat interface

### Management Endpoints
- `GET /health` - System health check
- `GET /intents` - List available intents
- `POST /reset-seats` - Reset available seats (admin)

### Example API Usage
```python
import requests

# Regular chat
response = requests.post("http://localhost:8000/chat", 
    json={"message": "Book a table for 2", "use_ensemble": True}
)
print(response.json())

# Health check
status = requests.get("http://localhost:8000/health")
print(status.json())
```

## 🧠 Model Architectures

### 1. Advanced Model (CNN + LSTM)
```python
Embedding → SpatialDropout → Multi-scale CNN → Bidirectional LSTM → Dense Layers
```
- **Embedding**: 300-dimensional word vectors
- **CNN**: Multi-scale feature extraction (2,3,4-grams)
- **LSTM**: Bidirectional context understanding
- **Regularization**: Dropout, BatchNorm, L2 regularization

### 2. Transformer Model
```python
Embedding → MultiHeadAttention → LayerNorm → FeedForward → Output
```
- **Self-Attention**: 8-head multi-head attention
- **Positional Encoding**: Implicit through embeddings
- **Layer Normalization**: Stabilized training

### 3. Ensemble Model
- **Weighted Average**: 40% advanced, 40% transformer, 20% CNN
- **Fallback Handling**: Automatic fallback to single model

## 📈 Performance Optimization

### Training Techniques
- **Learning Rate Scheduling**: Exponential decay
- **Early Stopping**: Prevents overfitting
- **Class Weighting**: Handles imbalanced data
- **Gradient Clipping**: Prevents exploding gradients

### Data Enhancement
- **Synonym Replacement**: Using WordNet thesaurus
- **Random Insertion**: Adds contextual synonyms
- **Data Balancing**: Upsamples minority classes

## 🔧 Configuration

### Hyperparameters
```python
best_config = {
    'embedding_dim': 300,      # Word vector dimensions
    'learning_rate': 0.001,    # Optimizer learning rate  
    'dropout_rate': 0.3,       # Dropout for regularization
    'batch_size': 32,          # Training batch size
    'epochs': 150              # Training epochs
}
```

### Model Settings
- **Vocabulary Size**: Dynamic based on training data
- **Sequence Length**: 95th percentile of training sequences
- **Embedding Dimensions**: 100-300 (tuned automatically)
- **Output Classes**: Based on intent tags

## 📊 Evaluation Metrics

The training process evaluates:
- **Accuracy**: Overall prediction correctness
- **Precision**: Relevant predictions among selected
- **Recall**: Relevant predictions captured
- **Confusion Matrix**: Visual classification performance
- **Training History**: Accuracy/loss over epochs

## 🐛 Troubleshooting

### Common Issues

**Model Loading Errors**
```bash
# Ensure all model files are present
ls filesTech/ensemble_models/
```

**NLTK Data Missing**
```python
import nltk
nltk.download('all')  # Download all required data
```

**Memory Issues**
- Reduce batch size in training
- Use smaller embedding dimensions
- Enable GPU acceleration

**Low Accuracy**
- Add more training data
- Increase model complexity
- Adjust hyperparameters

## 🚀 Deployment

### Production Deployment
```bash
# Install production server
pip install uvicorn

# Run production server
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

## 📝 Customization

### Adding New Intents
1. Add to `filesTech/final.json`:
```json
{
  "tag": "new_intent",
  "patterns": ["pattern1", "pattern2"],
  "responses": ["response1", "response2"]
}
```

2. Retrain models:
```bash
python jobs.py
```

### Modifying Business Logic
Edit `handle_special_cases()` in `app.py` for custom business rules.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow/Keras team for deep learning framework
- NLTK team for NLP tools
- FastAPI for modern web framework
- Contributors and testers

---

**Ready to build your intelligent restaurant chatbot?** Start with the installation guide above and explore the powerful features of this advanced NLP system!