import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample chatbot training data
training_data = [
    ("Hi", "greeting"),
    ("Hello", "greeting"),
    ("Hey", "greeting"),
    ("How are you?", "greeting"),
    ("Bye", "farewell"),
    ("Goodbye", "farewell"),
    ("See you later", "farewell"),
    ("Thanks", "thanks"),
    ("Thank you", "thanks"),
    ("What's your name?", "name_query"),
    ("Who are you?", "name_query"),
    ("What can you do?", "capability"),
    ("Tell me a joke", "joke"),
]

# Separate input sentences and labels
X = [x[0] for x in training_data]
y = [x[1] for x in training_data]

# ✅ Simple tokenizer (no NLTK dependency)
def simple_tokenizer(text):
    return re.findall(r'\b\w\w+\b', text.lower())

# Create a pipeline model (vectorizer + classifier)
model = Pipeline([
    ('vectorizer', CountVectorizer(tokenizer=simple_tokenizer)),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(X, y)

# Function to generate chatbot response
def chatbot_response(user_input):
    tag = model.predict([user_input])[0]
    responses = {
        "greeting": "Hello! How can I assist you today?",
        "farewell": "Goodbye! Have a great day!",
        "thanks": "You're welcome!",
        "name_query": "I'm your friendly chatbot!",
        "capability": "I can answer basic questions. Try asking me something!",
        "joke": "Why don't scientists trust atoms? Because they make up everything!"
    }
    return responses.get(tag, "Sorry, I didn’t understand that. Can you rephrase?")

# Example chat
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print("Bot:", response)
