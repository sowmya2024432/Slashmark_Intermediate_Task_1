# Slashmark_Intermediate_Task_1

🧠 Simple Chatbot with Scikit-learn
This is a basic rule-based chatbot built using Python and the Scikit-learn library. It uses a Naive Bayes classifier to categorize user inputs into predefined tags like greetings, farewells, and more.

📦 Features
Lightweight — No external NLP libraries (e.g., NLTK or spaCy)

Easy to train and extend with new intents

Uses CountVectorizer + MultinomialNB pipeline

Simple command-line interaction

🚀 Getting Started

Make sure you have Python 3.x installed.

Install required packages:

pip install scikit-learn

🧠 How It Works
The chatbot is trained with a small dataset of example inputs mapped to categories (tags). When a user enters a message, the model predicts its tag and selects a predefined response.

⚙️ Used Technologies
Python 3 – Core programming language

Scikit-learn – For text vectorization and Naive Bayes classification

Regular Expressions (re module) – Lightweight text tokenization

CountVectorizer – Converts text into numerical feature vectors

MultinomialNB – Naive Bayes classifier for multinomially distributed data

📜 License
This project is open-source and available under the MIT License.
