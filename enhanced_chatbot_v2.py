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
from textblob import TextBlob
from fuzzywuzzy import fuzz
from googletrans import Translator
import networkx as nx
from datetime import datetime
import random
from collections import deque
from typing import Dict, List, Tuple, Optional

class EnhancedChatbotV2:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        # Initialize conversation memory
        self.conversation_history = deque(maxlen=10)
        self.context = {}
        
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
        
        # Initialize translator
        self.translator = Translator()
        self.supported_languages = ['en', 'es', 'fr', 'de']
        
        # Initialize knowledge graph
        self.knowledge_graph = nx.DiGraph()
        
        # Load data files
        self.models_data = self._load_json('data/models.json')
        self.dealerships_data = self._load_json('data/dealerships.json')
        self.financing_data = self._load_json('data/financing.json')
        
        # Define intents and their patterns
        self.intents = {
            "general_inquiry": {
                "patterns": ["available", "latest", "models", "features", "compare", "price", "tell me about"],
                "entities": ["model", "feature", "specification"]
            },
            "customization_query": {
                "patterns": ["customize", "color", "option", "sunroof", "interior", "configure", "personalize"],
                "entities": ["model", "feature", "option_type"]
            },
            "pricing_finance_query": {
                "patterns": ["price", "cost", "finance", "lease", "interest", "payment", "trade-in"],
                "entities": ["model", "year", "variant", "finance_type"]
            },
            "test_drive_query": {
                "patterns": ["test drive", "dealership", "appointment", "visit", "tour", "showroom"],
                "entities": ["model", "location", "dealer_type"]
            },
            "performance_tech_query": {
                "patterns": ["speed", "performance", "motor", "electric", "hybrid", "safety", "technology"],
                "entities": ["model", "specification", "feature_type"]
            },
            "maintenance_warranty_query": {
                "patterns": ["warranty", "service", "maintenance", "repair", "center", "appointment"],
                "entities": ["model", "service_type", "warranty_type"]
            },
            "promotions_query": {
                "patterns": ["promotion", "discount", "offer", "deal", "loyalty", "program", "limited"],
                "entities": ["model", "promotion_type"]
            },
            "pre_owned_query": {
                "patterns": ["pre-owned", "used", "certified", "trade-in", "value", "second-hand"],
                "entities": ["model", "year", "condition"]
            },
            "delivery_query": {
                "patterns": ["delivery", "order", "track", "available", "purchase", "ship"],
                "entities": ["model", "delivery_type"]
            },
            "miscellaneous_query": {
                "patterns": ["experience", "museum", "factory", "merchandise", "fleet", "rent"],
                "entities": ["service_type", "location"]
            },
            "model_comparison_query": {
                "patterns": ["compare", "difference", "versus", "vs", "better", "between", "versus"],
                "entities": ["model1", "model2", "comparison_aspect"]
            },
            "competitor_comparison_query": {
                "patterns": ["versus", "vs", "better than", "compared to", "against", "versus"],
                "entities": ["model", "competitor", "comparison_aspect"]
            }
        }
        
        # Define entity types and their patterns
        self.entity_patterns = {
            "model": {
                "patterns": [
                    "911", "Cayenne", "Macan", "Panamera", "Taycan", "Boxster", "Cayman",
                    "Carrera", "Turbo", "GT3", "GT4", "RS", "Targa", "GTS", "S", "4S"
                ],
                "type": "model"
            },
            "year": {
                "patterns": [r"\b\d{4}\b"],
                "type": "year"
            },
            "variant": {
                "patterns": ["S", "GTS", "Turbo", "GT3", "GT4", "RS", "Targa", "4S", "Carrera"],
                "type": "variant"
            },
            "specification": {
                "patterns": [
                    "0-60", "top speed", "range", "battery", "engine", "horsepower",
                    "acceleration", "handling", "cornering", "AWD", "performance"
                ],
                "type": "specification"
            },
            "feature_type": {
                "patterns": [
                    "safety", "comfort", "technology", "infotainment", "driving",
                    "sunroof", "interior", "exterior", "color", "option"
                ],
                "type": "feature"
            },
            "service_type": {
                "patterns": [
                    "warranty", "service", "maintenance", "repair", "center",
                    "test drive", "delivery", "tracking", "appointment"
                ],
                "type": "service"
            },
            "location": {
                "patterns": [r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"],
                "type": "location"
            },
            "dealer_type": {
                "patterns": ["dealer", "dealership", "center", "store", "showroom"],
                "type": "dealer"
            },
            "finance_type": {
                "patterns": ["finance", "lease", "loan", "payment", "interest", "trade-in"],
                "type": "finance"
            },
            "warranty_type": {
                "patterns": ["warranty", "extended", "certified", "pre-owned", "limited"],
                "type": "warranty"
            },
            "promotion_type": {
                "patterns": ["promotion", "discount", "offer", "deal", "loyalty", "program"],
                "type": "promotion"
            },
            "delivery_type": {
                "patterns": ["delivery", "shipping", "tracking", "order", "purchase"],
                "type": "delivery"
            },
            "competitor": {
                "patterns": [
                    "Tesla", "BMW", "Mercedes", "Audi", "Lamborghini", "Ferrari",
                    "Range Rover", "Chevrolet", "Porsche"
                ],
                "type": "competitor"
            },
            "comparison_aspect": {
                "patterns": [
                    "performance", "speed", "handling", "price", "comfort",
                    "technology", "safety", "range", "acceleration"
                ],
                "type": "comparison"
            },
            "condition": {
                "patterns": ["new", "used", "pre-owned", "certified", "second-hand"],
                "type": "condition"
            }
        }
        
        # Initialize response templates
        self.response_templates = {
            "general_inquiry": {
                "model_info": "The {model} is {description}. Key features include {features}.",
                "model_list": "Currently available Porsche models include: {models}.",
                "latest_models": "The latest Porsche models include: {models} with {features}."
            },
            "customization_query": {
                "color_options": "The {model} is available in {colors}.",
                "feature_options": "You can add {features} to your {model}.",
                "interior_options": "Interior customization options for the {model} include {options}.",
                "online_config": "You can configure your {model} online using our configurator tool."
            },
            "pricing_finance_query": {
                "price_range": "The {model} starts at {base_price} and can go up to {max_price} with options.",
                "financing": "Porsche offers various financing options including {options}.",
                "lease": "Leasing options are available for the {model} starting at {lease_price} per month.",
                "trade_in": "Porsche offers a trade-in program. The value depends on your current vehicle."
            },
            "test_drive_query": {
                "booking": "You can book a test drive for the {model} through {booking_method}.",
                "dealership": "The nearest Porsche dealership is {location}.",
                "appointment": "Appointments are recommended for {service_type}.",
                "virtual_tour": "Virtual tours are available through {platform}."
            },
            "performance_tech_query": {
                "performance": "The {model} achieves {specifications}.",
                "technology": "The {model} features {features}.",
                "safety": "Safety features include {features}.",
                "connectivity": "The infotainment system supports {features}."
            },
            "maintenance_warranty_query": {
                "warranty": "Porsche's warranty includes {coverage}.",
                "service": "Recommended service intervals are {intervals}.",
                "service_center": "Authorized service centers are located at {locations}.",
                "appointment": "You can book service appointments {booking_method}."
            },
            "promotions_query": {
                "current_offers": "Current promotions include {offers}.",
                "loyalty_program": "Our loyalty program offers {benefits}.",
                "limited_edition": "Limited edition models include {models}.",
                "seasonal": "Current seasonal offers include {offers}."
            },
            "pre_owned_query": {
                "availability": "Certified pre-owned {model} vehicles are available {availability}.",
                "warranty": "Pre-owned vehicles come with {warranty_details}.",
                "certification": "Our certification process includes {process}.",
                "value": "Trade-in values depend on {factors}."
            },
            "delivery_query": {
                "timeline": "Custom {model} delivery typically takes {time}.",
                "tracking": "You can track your order through {method}.",
                "delivery_options": "Delivery options include {options}.",
                "charges": "Delivery charges vary based on {factors}."
            },
            "miscellaneous_query": {
                "experience": "Porsche offers {experiences}.",
                "museum": "The Porsche museum is located at {location}.",
                "merchandise": "Porsche merchandise is available {availability}.",
                "fleet": "Corporate fleet programs offer {benefits}."
            },
            "model_comparison_query": {
                "comparison": "When comparing {model1} and {model2}, {differences}.",
                "differences": "Key differences include {differences}.",
                "recommendation": "For {use_case}, the {model} would be recommended."
            },
            "competitor_comparison_query": {
                "vs_competitor": "Compared to the {competitor}, the {model} offers {advantages}.",
                "advantages": "Key advantages include {advantages}.",
                "performance": "In terms of {aspect}, the {model} {comparison}."
            }
        }
        
        # Initialize learning data
        self.successful_patterns = {}
        self.failed_patterns = {}

    def _load_json(self, filepath: str) -> Dict:
        """Load JSON data from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found")
            return {}
        except json.JSONDecodeError:
            print(f"Warning: Error decoding {filepath}")
            return {}

    def _get_model_info(self, model: str) -> Dict:
        """Get information about a specific model"""
        return self.models_data.get(model, {})

    def _get_dealership_info(self, location: str = None) -> Dict:
        """Get dealership information"""
        if location:
            for dealer in self.dealerships_data.get('dealerships', []):
                if location.lower() in dealer['location'].lower():
                    return dealer
        return self.dealerships_data.get('dealerships', [{}])[0]

    def _get_financing_info(self, model: str = None) -> Dict:
        """Get financing information"""
        if model:
            return {
                'lease': self.financing_data.get('financing_options', {}).get('lease', {}).get('monthly_payments', {}).get(model, {}),
                'purchase': self.financing_data.get('financing_options', {}).get('purchase', {})
            }
        return self.financing_data.get('financing_options', {})

    def _get_promotions(self, model: str = None) -> List[Dict]:
        """Get current promotions"""
        promotions = self.financing_data.get('promotions', {}).get('current_offers', [])
        if model:
            return [p for p in promotions if p.get('model') == model]
        return promotions

    def _get_warranty_info(self, warranty_type: str = 'new_vehicle') -> Dict:
        """Get warranty information"""
        return self.dealerships_data.get('warranty', {}).get(warranty_type, {})

    def _get_test_drive_info(self) -> Dict:
        """Get test drive information"""
        return self.dealerships_data.get('test_drive', {})

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better matching"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of input text"""
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def _get_recent_context(self) -> Dict:
        """Get recent conversation context"""
        return {
            'history': list(self.conversation_history),
            'context': self.context
        }

    def _query_knowledge_graph(self, entities: List[str]) -> Optional[str]:
        """Query knowledge graph for related information"""
        related_info = []
        for entity in entities:
            if entity in self.knowledge_graph:
                neighbors = self.knowledge_graph.neighbors(entity)
                for neighbor in neighbors:
                    edge_data = self.knowledge_graph.get_edge_data(entity, neighbor)
                    related_info.append(f"{entity} is related to {neighbor}: {edge_data.get('relation', '')}")
        return "\n".join(related_info) if related_info else None

    def _get_response_confidence(self, query: str, best_match: str, similarity: float) -> float:
        """Calculate confidence score for response"""
        confidence = {
            'bert_similarity': similarity,
            'bow_similarity': self._get_bow_similarity(query)[0],
            'entity_match': len(self._extract_entities(query)) > 0,
            'context_relevance': self._check_context_relevance(query)
        }
        return sum(confidence.values()) / len(confidence)

    def _check_context_relevance(self, query: str) -> float:
        """Check if query is relevant to recent context"""
        if not self.conversation_history:
            return 0.5
        recent_context = " ".join([item.get('user', '') for item in self.conversation_history])
        return fuzz.ratio(query.lower(), recent_context.lower()) / 100.0

    def _handle_ambiguous_query(self, query: str) -> Dict[str, any]:
        """Handle ambiguous queries with clarifying questions"""
        clarifying_questions = [
            "Could you specify which model you're interested in?",
            "Are you asking about price, performance, or something else?",
            "Would you like to compare specific models?",
            "Are you interested in new or pre-owned vehicles?",
            "Would you like information about specific features?"
        ]
        return {
            "response": "I'm not sure I understand completely. " + random.choice(clarifying_questions),
            "needs_clarification": True
        }

    def _learn_from_interaction(self, query: str, response: str, user_feedback: str) -> None:
        """Learn from user feedback"""
        if user_feedback == 'helpful':
            self.successful_patterns[query] = response
        elif user_feedback == 'not_helpful':
            self.failed_patterns[query] = response

    def _fill_template(self, template_key: str, variables: Dict[str, str]) -> str:
        """Fill response template with variables"""
        return self.response_templates[template_key].format(**variables)

    def _generate_dynamic_response(self, base_response: str, query: str, context: Dict) -> str:
        """Generate dynamic response based on context"""
        if "price" in query.lower():
            current_year = datetime.now().year
            base_response = base_response.replace("$", f"${current_year} ")
        
        if context.get("user_name"):
            base_response = f"Hi {context['user_name']}, {base_response.lower()}"
        
        return base_response

    def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        try:
            return self.translator.detect(text).lang
        except:
            return 'en'

    def _create_bow_features(self) -> None:
        """Create Bag of Words features for all questions"""
        processed_questions = [self._preprocess_text(q) for q in self.qa_data.keys()]
        return self.tfidf_vectorizer.fit_transform(processed_questions)

    def _get_bow_similarity(self, query: str) -> np.ndarray:
        """Get similarity score using Bag of Words"""
        processed_query = self._preprocess_text(query)
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        return cosine_similarity(query_vector, self.bow_features)[0]

    def _get_bert_embedding(self, text: str) -> torch.Tensor:
        """Get BERT embedding for input text"""
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            
        # Use the [CLS] token embedding as the sentence representation
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Convert to numpy array for similarity calculation
        return embeddings.numpy()

    def _find_best_match(self, query_embedding: torch.Tensor, query: str) -> Tuple[str, float]:
        """Find best matching question using combined similarity scores"""
        best_similarity = -1
        best_match = None
        
        bert_similarities = []
        for question, embedding in self.question_embeddings.items():
            similarity = cosine_similarity(query_embedding, embedding)[0][0]
            bert_similarities.append(similarity)
        
        bow_similarities = self._get_bow_similarity(query)
        
        fuzzy_scores = []
        for question in self.qa_data.keys():
            score = fuzz.ratio(query.lower(), question.lower())
            fuzzy_scores.append(score / 100.0)
        
        questions = list(self.qa_data.keys())
        for i in range(len(questions)):
            combined_similarity = (
                0.5 * bert_similarities[i] + 
                0.2 * bow_similarities[i] + 
                0.3 * fuzzy_scores[i]
            )
            if combined_similarity > best_similarity:
                best_similarity = combined_similarity
                best_match = questions[i]
        
        return best_match, best_similarity

    def detect_intent(self, text: str) -> Tuple[str, float]:
        """Detect the intent of the user's query"""
        text = text.lower()
        best_intent = "general_inquiry"
        best_confidence = 0.0
        
        for intent_name, intent_data in self.intents.items():
            pattern_matches = sum(1 for pattern in intent_data["patterns"] 
                                if pattern in text)
            confidence = pattern_matches / len(intent_data["patterns"])
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_intent = intent_name
        
        return best_intent, best_confidence

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract entities from the text using both spaCy and pattern matching"""
        entities = []
        
        # Use spaCy for general entity extraction
        doc = self.nlp(text)
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        # Use pattern matching for specific entity types
        for entity_type, entity_data in self.entity_patterns.items():
            for pattern in entity_data["patterns"]:
                if isinstance(pattern, str):
                    if pattern.lower() in text.lower():
                        start = text.lower().find(pattern.lower())
                        entities.append({
                            "text": text[start:start + len(pattern)],
                            "type": entity_data["type"],
                            "start": start,
                            "end": start + len(pattern)
                        })
                else:
                    matches = pattern.finditer(text)
                    for match in matches:
                        entities.append({
                            "text": match.group(),
                            "type": entity_data["type"],
                            "start": match.start(),
                            "end": match.end()
                        })
        
        # Remove overlapping entities
        entities.sort(key=lambda x: x["start"])
        filtered_entities = []
        for i, entity in enumerate(entities):
            if i == 0 or entity["start"] >= entities[i-1]["end"]:
                filtered_entities.append(entity)
        
        return filtered_entities

    def get_response(self, user_input: str, user_name: Optional[str] = None, 
                    user_feedback: Optional[str] = None) -> Dict[str, any]:
        """Get response for user input with all enhancements"""
        # Update context
        if user_name:
            self.context['user_name'] = user_name
        
        # Detect language
        detected_lang = self.detect_language(user_input)
        if detected_lang != 'en':
            user_input = self.translator.translate(user_input, dest='en').text
        
        # Add to conversation history
        self.conversation_history.append({"user": user_input})
        
        # Get recent context
        recent_context = self._get_recent_context()
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment(user_input)
        
        # Detect intent and extract entities
        intent, intent_confidence = self.detect_intent(user_input)
        entities = self.extract_entities(user_input)
        
        # Get BERT embedding
        query_embedding = self._get_bert_embedding(user_input)
        
        # Find best match
        best_match, similarity = self._find_best_match(query_embedding, user_input)
        
        # Calculate confidence
        confidence = self._get_response_confidence(user_input, best_match, similarity)
        
        # Handle ambiguous queries
        if confidence < 0.4 or intent_confidence < 0.3:
            return self._handle_ambiguous_query(user_input)
        
        # Get base response
        if similarity < 0.5:
            response = "I apologize, but I don't have specific information about that. Could you please rephrase your question or ask about something else related to Porsche vehicles?"
            related_questions = [
                "What models does Porsche offer?",
                "What is the price range?",
                "Where can I find a dealership?"
            ]
        else:
            response = self.qa_data[best_match]["answer"]
            related_questions = self.qa_data[best_match]["related_questions"]
        
        # Enhance response with knowledge graph and entities
        if entities:
            knowledge_info = self._query_knowledge_graph([e["text"] for e in entities])
            if knowledge_info:
                response += f"\n\nRelated Information:\n{knowledge_info}"
        
        # Generate dynamic response
        response = self._generate_dynamic_response(response, user_input, self.context)
        
        # Add sentiment-based tone
        if sentiment < -0.2:
            response = f"I understand your concern. {response}"
        elif sentiment > 0.2:
            response = f"I'm glad you're interested! {response}"
        
        # Add confidence-based disclaimer if needed
        if confidence < 0.6:
            response = "I'm not entirely sure about that, but based on what I know: " + response
        
        # Translate response if needed
        if detected_lang != 'en':
            response = self.translator.translate(response, dest=detected_lang).text
        
        # Learn from feedback if provided
        if user_feedback:
            self._learn_from_interaction(user_input, response, user_feedback)
        
        # Add bot response to history
        self.conversation_history.append({"bot": response})
        
        return {
            "response": response,
            "related_questions": related_questions,
            "confidence": confidence,
            "needs_clarification": False,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "entities": entities
        } 