import spacy
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

class NLPChatbot:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        
        # Load custom dataset from JSON files
        self.qa_data = self._load_qa_data()
        
        # Create embeddings and BoW features
        self.question_embeddings = self._create_embeddings()
        self.bow_features = self._create_bow_features()
        
    def _preprocess_text(self, text):
        """Preprocess text for better matching"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        return ' '.join(tokens)
    
    def _create_bow_features(self):
        """Create Bag of Words features for all questions"""
        # Preprocess all questions
        processed_questions = [self._preprocess_text(q) for q in self.qa_data.keys()]
        
        # Fit and transform the questions
        bow_features = self.tfidf_vectorizer.fit_transform(processed_questions)
        
        return bow_features
    
    def _get_bow_similarity(self, query):
        """Get similarity score using Bag of Words"""
        # Preprocess the query
        processed_query = self._preprocess_text(query)
        
        # Transform the query
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Calculate similarities with all questions
        similarities = cosine_similarity(query_vector, self.bow_features)[0]
        
        return similarities
    
    def _load_qa_data(self):
        """Load the custom Q&A dataset from JSON files"""
        qa_data = {}
        data_dir = "data"
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created {data_dir} directory. Please add your JSON files there.")
            return {}
        
        # Load all JSON files from the data directory
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Merge the data from each file
                        qa_data.update(data)
                except json.JSONDecodeError as e:
                    print(f"Error reading {filename}: {e}")
                except Exception as e:
                    print(f"Unexpected error reading {filename}: {e}")
        
        if not qa_data:
            print("No valid JSON data found. Please add your JSON files to the data directory.")
            return {}
            
        return qa_data
    
    def _create_embeddings(self):
        """Create BERT embeddings for all questions in the dataset"""
        embeddings = {}
        for question in self.qa_data.keys():
            inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            embeddings[question] = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings
    
    def _get_bert_embedding(self, text):
        """Get BERT embedding for input text"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    
    def _find_best_match(self, query_embedding, query):
        """Find the best matching question using combined similarity scores"""
        best_similarity = -1
        best_match = None
        
        # Get BERT similarities
        bert_similarities = []
        for question, embedding in self.question_embeddings.items():
            similarity = cosine_similarity(query_embedding, embedding)[0][0]
            bert_similarities.append(similarity)
        
        # Get BoW similarities
        bow_similarities = self._get_bow_similarity(query)
        
        # Combine similarities (weighted average)
        questions = list(self.qa_data.keys())
        for i in range(len(questions)):
            combined_similarity = 0.7 * bert_similarities[i] + 0.3 * bow_similarities[i]
            if combined_similarity > best_similarity:
                best_similarity = combined_similarity
                best_match = questions[i]
        
        return best_match, best_similarity
    
    def process_query(self, query):
        """Process user query and return appropriate response"""
        if not self.qa_data:
            return {
                "response": "I apologize, but I don't have any data loaded. Please make sure your JSON files are properly placed in the data directory.",
                "related_questions": []
            }

        query_embedding = self._get_bert_embedding(query)
        best_match, similarity = self._find_best_match(query_embedding, query)
        
        if similarity < 0.5:
            return {
                "response": "I apologize, but I don't have specific information about that. Could you please rephrase your question or ask about something else related to Porsche vehicles?",
                "related_questions": [
                    "What models does Porsche offer?",
                    "What is the price range?",
                    "Where can I find a dealership?"
                ]
            }
        
        return {
            "response": self.qa_data[best_match]["answer"],
            "related_questions": self.qa_data[best_match]["related_questions"]
        }
    
    def get_response(self, user_input):
        """Get response for user input with additional NLP processing"""
        doc = self.nlp(user_input.lower())
        entities = [ent.text for ent in doc.ents]
        key_terms = [token.text for token in doc if not token.is_stop and token.is_alpha]
        
        result = self.process_query(user_input)
        response = result["response"]
        
        if entities:
            response = f"Regarding {', '.join(entities)}, {response.lower()}"
        
        return {
            "response": response,
            "related_questions": result["related_questions"]
        } 