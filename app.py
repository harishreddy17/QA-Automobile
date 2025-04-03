from flask import Flask, render_template, request, jsonify, session
from enhanced_chatbot import EnhancedChatbot
import time

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session management
chatbot = EnhancedChatbot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    user_name = session.get('user_name')
    user_feedback = request.json.get('feedback')
    
    # Get response from enhanced chatbot
    result = chatbot.get_response(
        user_message,
        user_name=user_name,
        user_feedback=user_feedback
    )
    
    # Simulate typing delay for more natural feel
    time.sleep(0.5)
    
    return jsonify({
        'response': result['response'],
        'related_questions': result['related_questions'],
        'confidence': result['confidence'],
        'needs_clarification': result['needs_clarification'],
        'timestamp': time.strftime('%H:%M:%S'),
        'intent': result['intent'],
        'intent_confidence': result['intent_confidence'],
        'entities': result['entities']
    })

@app.route('/set_name', methods=['POST'])
def set_name():
    name = request.json.get('name', '')
    if name:
        session['user_name'] = name
        return jsonify({'success': True})
    return jsonify({'success': False})

if __name__ == '__main__':
    app.run(debug=True) 