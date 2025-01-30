import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import webbrowser
import datetime

class Chatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.load_chatbot_data()
        self.conversation_history = []

    def load_chatbot_data(self):
        with open(r"D:\\project101\\cutomer service\\intents.json", "r") as f:
            self.intents = json.load(f)
        with open("D:\project101\cutomer service\words.pkl", "rb") as f:
            self.words = pickle.load(f)
        with open("D:\project101\cutomer service\classes.pkl", "rb") as f:
            self.classes = pickle.load(f)
        self.model = load_model("D:\project101\cutomer service\chatbot_model.h5")

    def clean_up_sentence(self, sentence):
        return [
            self.lemmatizer.lemmatize(word.lower())
            for word in nltk.word_tokenize(sentence)
        ]

    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [1 if word in sentence_words else 0 for word in self.words]
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return [
            {"intent": self.classes[r[0]], "probability": str(r[1])} for r in results
        ]

    def get_response(self, intents_list):
        if not intents_list:
            return "I'm not sure how to respond to that. Can you please rephrase your question?"
        tag = intents_list[0]["intent"]
        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        return "I'm sorry, I don't have a specific response for that. Can you try asking something else?"

    def chatbot_response(self, user_message):
        if user_message.lower() in ["exit", "quit", "bye"]:
            return "Goodbye, it was nice chatting with you."
        elif user_message.lower().startswith("search "):
            query = user_message[7:]
            webbrowser.open(f"https://www.google.com/search?q={query}")
            return f"I've opened a web search for '{query}'."
        elif user_message.lower() == "time":
            return f"Current time is {datetime.datetime.now().strftime('%H:%M:%S')}"
        else:
            intents = self.predict_class(user_message)
            return self.get_response(intents)


# Initialize the chatbot
chatbot = Chatbot()

# Streamlit UI
st.set_page_config(page_title="GYM-BOT", page_icon="💬")
st.title("GYM-BOT")

# Display chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Display existing conversation
for message in st.session_state['chat_history']:
    st.markdown(f"**{message['role']}**: {message['text']}")

# Input text box for the user
user_input = st.text_input("You:", "")

# Button to submit user input
if st.button("Send") and user_input:
    # Show user message
    st.session_state['chat_history'].append({"role": "You", "text": user_input})
    
    # Get bot response
    bot_response = chatbot.chatbot_response(user_input)
    st.session_state['chat_history'].append({"role": "Bot", "text": bot_response})
    
    # Update the conversation
    st.rerun()

# Clear chat history
if st.button("Clear Chat"):
    st.session_state['chat_history'] = []

# Save chat history
if st.button("Save Chat"):
    filename = f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w") as f:
        for message in st.session_state['chat_history']:
            f.write(f"{message['role']}: {message['text']}\n")
    st.success(f"Chat history saved as {filename}")

# Show help text
if st.button("Help"):
    help_text = """
    **Special Commands**:
    - Type 'exit', 'quit', or 'bye' to end the conversation.
    - Type 'search <query>' to open a web search.
    - Type 'time' to get the current time.

    **Features**:
    - Clear Chat: Clears the current conversation.
    - Save Chat: Saves the conversation history to a file.
    - Help: Shows this help message.

    Enjoy chatting!
    """
    st.info(help_text)
