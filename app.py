# Imports
import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import json
import pickle
import re
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional
import asyncio

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Define the preprocessor class
class AdvancedTextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vocab = set()
        self.word2idx = {}
        self.idx2word = {}
        self.max_sequence_length = 0
        
    def clean_text(self, text):
        """Advanced text cleaning and normalization"""
        if not text:
            return ""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits but keep basic punctuation
        text = re.sub(r'[^a-zA-Z\s\?\!\.\,]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def advanced_tokenize(self, text):
        """Advanced tokenization with POS tagging"""
        if not text:
            return []
            
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 1]
        
        # Lemmatization with POS tagging
        try:
            pos_tags = nltk.pos_tag(tokens)
            lemmatized_tokens = []
            
            for token, pos in pos_tags:
                if pos.startswith('V'):  # Verb
                    lemma = self.lemmatizer.lemmatize(token, pos='v')
                elif pos.startswith('J'):  # Adjective
                    lemma = self.lemmatizer.lemmatize(token, pos='a')
                elif pos.startswith('R'):  # Adverb
                    lemma = self.lemmatizer.lemmatize(token, pos='r')
                else:  # Noun and others
                    lemma = self.lemmatizer.lemmatize(token)
                lemmatized_tokens.append(lemma)
            
            return lemmatized_tokens
        except:
            # Fallback to simple lemmatization if POS tagging fails
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
        self.word2idx = {word: idx + 1 for idx, word in enumerate(self.vocab)}  # 0 reserved for padding
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        # Calculate max sequence length
        sequence_lengths = [len(self.advanced_tokenize(self.clean_text(doc))) for doc in documents]
        if sequence_lengths:
            self.max_sequence_length = max(5, int(np.percentile(sequence_lengths, 95)))
        else:
            self.max_sequence_length = 20
        
        return all_tokens
    
    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        cleaned_text = self.clean_text(text)
        tokens = self.advanced_tokenize(cleaned_text)
        sequence = [self.word2idx.get(token, 0) for token in tokens]  # 0 for OOV
        return sequence

# Initialize FastAPI app
app = FastAPI(title="Restaurant Chatbot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables
seat_count = 50

# Load data and model
try:
    # Load preprocessor and model artifacts
    with open("filesTech/preprocessor.pickle", "rb") as f:
        artifacts = pickle.load(f)
    
    preprocessor = artifacts['preprocessor']
    label2idx = artifacts['label2idx']
    idx2label = artifacts['idx2label']
    max_sequence_length = artifacts['max_sequence_length']
    vocab_size = artifacts['vocab_size']
    
    # Load the trained model
    model = load_model("filesTech/advanced_model.h5")
    
    # Load intents data
    with open("filesTech/final.json", encoding='utf-8') as file:
        data = json.load(file)
    
    print("✅ Model and data loaded successfully!")
    print(f"✅ Vocabulary size: {vocab_size}")
    print(f"✅ Number of classes: {len(label2idx)}")
    
except Exception as e:
    print(f"❌ Error loading model or data: {e}")
    print("🔄 Creating a simple fallback model...")
    # Create a simple fallback
    preprocessor = AdvancedTextPreprocessor()
    label2idx = {"greeting": 0, "fallback": 1}
    idx2label = {0: "greeting", 1: "fallback"}
    max_sequence_length = 20
    vocab_size = 100
    model = None
    data = {"intents": []}

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float

def format_response(response):
    """Add proper spacing to responses that are missing spaces"""
    if not response:
        return response
    
    # General formatting rules
    fixes = [
        (r'([a-z])([A-Z][a-z])', r'\1 \2'),
        (r'(\d)(am|pm|AM|PM)', r'\1 \2'),
        (r'([a-zA-Z])-([a-zA-Z])', r'\1 - \2'),
        (r'(\dam|\dpm|\dAM|\dPM)-(\dam|\dpm|\dAM|\dPM)', r'\1 - \2'),
        (r'([.!?])([A-Z])', r'\1 \2'),
        (r'\s+', ' '),
    ]
    
    formatted = response
    for pattern, replacement in fixes:
        formatted = re.sub(pattern, replacement, formatted)
    
    return formatted.strip()

def get_bot_response(message: str) -> dict:
    """Get bot response using the RNN model"""
    global seat_count
    
    if not message or not message.strip():
        return {
            "response": "I didn't receive any message. Please try again.",
            "intent": "unknown",
            "confidence": 0.0
        }
    
    try:
        # If model is not loaded, use fallback responses
        if model is None:
            return fallback_response(message)
        
        # Preprocess the input message
        sequence = preprocessor.text_to_sequence(message)
        padded_sequence = pad_sequences(
            [sequence], 
            maxlen=max_sequence_length, 
            padding='post', 
            truncating='post'
        )
        
        # Get prediction
        prediction = model.predict(padded_sequence, verbose=0)[0]
        predicted_idx = np.argmax(prediction)
        confidence = prediction[predicted_idx]
        intent = idx2label[predicted_idx]
        
        # Handle special business logic
        response_text = handle_special_cases(message, intent)
        if response_text:
            return {
                "response": format_response(response_text),
                "intent": intent,
                "confidence": float(confidence)
            }
        
        # Get response from intents data
        for intent_data in data['intents']:
            if intent_data['tag'] == intent:
                responses = intent_data['responses']
                response_text = random.choice(responses) if responses else "I'm still learning about this topic."
                break
        else:
            response_text = "I didn't quite get that, please try again."
        
        return {
            "response": format_response(response_text),
            "intent": intent,
            "confidence": float(confidence)
        }
        
    except Exception as e:
        print(f"Error in get_bot_response: {e}")
        return fallback_response(message)

def handle_special_cases(message: str, intent: str) -> str:
    """Handle special business logic cases"""
    global seat_count
    message_lower = message.lower()
    
    # Reservation availability check
    if "reservation" in message_lower and "available" in message_lower:
        return f"We have {seat_count} seats available for reservation."
    
    # Make reservation
    elif "reservation" in message_lower and any(word in message_lower for word in ["book", "reserve", "make"]):
        if seat_count > 0:
            seat_count -= 1
            return f"Reservation confirmed! We have {seat_count} seats remaining."
        else:
            return "Sorry, we're fully booked for today."
    
    return ""

def fallback_response(message: str) -> dict:
    """Fallback response when model is not available"""
    message_lower = message.lower()
    
    # Simple keyword-based fallback
    if any(word in message_lower for word in ["hello", "hi", "hey"]):
        return {
            "response": "Hello! Welcome to our restaurant. How can I help you?",
            "intent": "greeting",
            "confidence": 0.9
        }
    elif "reservation" in message_lower:
        return {
            "response": f"We have {seat_count} seats available. Would you like to book a table?",
            "intent": "book_table",
            "confidence": 0.8
        }
    elif "menu" in message_lower:
        return {
            "response": "We serve Italian, Chinese, Indian, and Continental cuisine. What would you like to know about?",
            "intent": "menu",
            "confidence": 0.8
        }
    elif "hour" in message_lower:
        return {
            "response": "We are open 10am-12am Monday-Friday, and 9am-1am on weekends!",
            "intent": "hours",
            "confidence": 0.9
        }
    else:
        return {
            "response": "I'm here to help with reservations, menu information, and restaurant details. What would you like to know?",
            "intent": "fallback",
            "confidence": 0.5
        }

# Routes
@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the chat interface"""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except:
        return HTMLResponse(content="<h1>Chatbot is running!</h1><p>Add your HTML template in templates/index.html</p>")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint for regular responses"""
    try:
        result = get_bot_response(request.message)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream")
async def stream_bot_response(message: str):
    """Stream bot response with Server-Sent Events"""
    async def generate():
        if not message:
            yield f"data: I didn't receive any message. Please try again.\n\n"
            yield "event: end\ndata: \n\n"
            return
        
        try:
            # Get the full response
            result = get_bot_response(message)
            full_response = result["response"]
            
            # Stream the response character by character
            for char in full_response:
                yield f"data: {char}\n\n"
                await asyncio.sleep(0.03)
            
            # Send end event with metadata
            yield f"event: end\ndata: {json.dumps({'intent': result['intent'], 'confidence': result['confidence']})}\n\n"
            
        except Exception as e:
            yield f"data: Sorry, I encountered an error: {str(e)}\n\n"
            yield "event: end\ndata: \n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "seat_count": seat_count,
        "model_loaded": model is not None,
        "intents_count": len(label2idx)
    }

@app.get("/intents")
async def list_intents():
    """List all available intents"""
    return {
        "intents": list(label2idx.keys()),
        "total_intents": len(label2idx)
    }

@app.post("/reset-seats")
async def reset_seats(count: int = 50):
    """Reset available seats (admin endpoint)"""
    global seat_count
    seat_count = count
    return {"message": f"Seat count reset to {seat_count}", "seat_count": seat_count}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )