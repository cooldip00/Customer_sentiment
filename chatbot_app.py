import streamlit as st
from transformers import pipeline
import re

# --- 1. Load Your Fine-Tuned Model (with Caching) ---
# @st.cache_resource is a powerful Streamlit feature. It ensures the model is
# loaded only once when the app starts, not on every user interaction, making it much faster.
@st.cache_resource
def load_chatbot_model():
    """Loads the fine-tuned T5 model and tokenizer."""
    model_path = "./final_retention_strategy_model"
    try:
        chatbot = pipeline("text2text-generation", model=model_path, tokenizer=model_path)
        print("Fine-tuned model loaded successfully.")
        return chatbot
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

chatbot_generator = load_chatbot_model()

# --- 2. Define the Chatbot Logic (Same as before) ---
# We keep the same keyword-based logic for asking clarifying questions.
category_keywords = {
    'Debit Card': ['debit', 'card', 'atm', 'pin', 'transaction'],
    'Credit Card': ['credit', 'card', 'bill', 'statement', 'rewards', 'limit'],
    'Loans': ['loan', 'emi', 'prepayment', 'application', 'disbursement'],
    'Account': ['account', 'savings', 'current', 'balance', 'statement', 'passbook'],
    'Branch Service': ['branch', 'staff', 'crowd'],
    'Mobile Banking': ['app', 'mobile', 'login', 'crash'],
    'Online Banking': ['online', 'website', 'netbanking']
}

def has_keywords(query):
    for _, keywords in category_keywords.items():
        if any(re.search(r'\b' + keyword + r'\b', query, re.IGNORECASE) for keyword in keywords):
            return True
    return False

def get_chatbot_response(user_query):
    """Generates a response using the fine-tuned model."""
    if chatbot_generator is None:
        return "Error: Model could not be loaded."
        
    if not user_query or len(user_query.strip()) < 10 or not has_keywords(user_query):
        return (
            "I'm sorry, I'm not quite sure how to help. Could you please provide "
            "more details about your issue? For example, is it related to a "
            "'loan', 'debit card', or your 'account'?"
        )

    # Craft the prompt for your T5 model
    input_prompt = (
        f"instruction: Act as a helpful and empathetic bank chatbot. "
        f"A customer has the following query: '{user_query}'. "
        f"Analyze the query and provide a direct, helpful response, and if applicable, "
        f"state which team it will be routed to."
    )
    
    # Generate the response with improved parameters
    try:
        generated_output = chatbot_generator(
            input_prompt, 
            max_length=150,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        return generated_output[0]['generated_text']
    except Exception as e:
        return f"Sorry, an error occurred: {e}"

# --- 3. Build the Streamlit App Interface ---
st.set_page_config(page_title="XYZ Bank Chatbot", layout="centered")

st.title("🤖 XYZ Bank Support Chatbot")
st.write("Powered by our custom fine-tuned model to provide helpful, relevant responses.")

# Initialize chat history in Streamlit's session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display the past messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input from a chat input box
user_input = st.chat_input("What is your question or feedback?")

if user_input:
    # Add user message to chat history and display it
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display the chatbot's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            bot_response = get_chatbot_response(user_input)
            st.markdown(bot_response)
    
    # Add bot response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})