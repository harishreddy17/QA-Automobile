# Automobile QA Chatbot

An advanced QA chatbot for automobile company queries using NLP techniques. This chatbot uses spaCy for natural language processing and BERT for understanding context and generating responses.

## Features

- Natural Language Processing using spaCy
- Context understanding using BERT
- Custom dataset for automobile-specific queries
- Interactive web interface
- Human-like responses

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Project Structure

- `app.py`: Main Flask application
- `nlp_processor.py`: NLP processing module
- `data/`: Contains the custom dataset
- `templates/`: HTML templates
- `static/`: CSS and JavaScript files 