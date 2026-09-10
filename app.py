import gradio as gr
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------------
# Intelligent Customer Support Chatbot
# ---------------------------------------------------------

# Training data
data = {
    "text": [
        "Where is my order?", "Can you track my order?",
        "What is my delivery status?", "When will my package arrive?",
        "How can I track my shipment?",

        "I want to cancel my order", "Please cancel my purchase",
        "Can I cancel my order?", "I ordered something by mistake",
        "How do I cancel an order?",

        "I want a refund", "How can I get my money back?",
        "I need to request a refund", "Can I get a refund for my order?",
        "Please help me with a refund",

        "My product is damaged", "I received a damaged item",
        "The product I received is broken", "I got a defective product",
        "What should I do if my product is damaged?",

        "What payment methods do you accept?", "Can I pay using UPI?",
        "Do you accept credit cards?", "Can I pay with a debit card?",
        "What are the available payment options?",

        "I forgot my password", "How can I reset my password?",
        "I cannot login to my account", "Help me change my password",
        "I am unable to access my account",

        "Hello", "Hi", "Hey", "Good morning", "Good evening",

        "Thank you", "Thanks for your help", "That was helpful",
        "Thank you very much", "Thanks",

        "I have a problem", "I need help", "Can you help me?",
        "I have an issue", "I need customer support"
    ],
    "intent": [
        "order_tracking", "order_tracking", "order_tracking",
        "order_tracking", "order_tracking",

        "cancel_order", "cancel_order", "cancel_order",
        "cancel_order", "cancel_order",

        "refund", "refund", "refund", "refund", "refund",

        "damaged_product", "damaged_product", "damaged_product",
        "damaged_product", "damaged_product",

        "payment", "payment", "payment", "payment", "payment",

        "password", "password", "password", "password", "password",

        "greeting", "greeting", "greeting", "greeting", "greeting",

        "thanks", "thanks", "thanks", "thanks", "thanks",

        "general_help", "general_help", "general_help",
        "general_help", "general_help"
    ]
}

df = pd.DataFrame(data)

# Load pretrained Transformer
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()[0]

# Create embeddings and train intent classifier
X_embeddings = np.array([get_embedding(text) for text in df["text"]])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df["intent"])

X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

responses = {
    "order_tracking":
        "📦 You can track your order using the tracking ID provided in your order confirmation.",

    "cancel_order":
        "❌ You can request order cancellation if the order has not been shipped yet.",

    "refund":
        "💰 You can request a refund through the orders section. Refund processing depends on the payment method.",

    "damaged_product":
        "🛠️ I'm sorry about that. Please provide your order details and contact customer support for a replacement or refund.",

    "payment":
        "💳 We support common payment methods such as UPI, debit cards and credit cards.",

    "password":
        "🔐 You can reset your password using the 'Forgot Password' option on the login page.",

    "greeting":
        "👋 Hello! Welcome to our customer support. How can I help you today?",

    "thanks":
        "😊 You're welcome! I'm happy to help.",

    "general_help":
        "💬 Sure! Please tell me more about the problem you are facing."
}

def chatbot(user_text):
    if not user_text or not user_text.strip():
        return "Please type a question so I can help you.", "unknown", 0.0

    embedding = get_embedding(user_text)
    prediction = classifier.predict([embedding])[0]
    intent = label_encoder.inverse_transform([prediction])[0]
    confidence = float(max(classifier.predict_proba([embedding])[0]))

    if confidence < 0.40:
        return (
            "I'm not completely sure I understand your question. "
            "Please explain the problem in a little more detail or contact customer support.",
            "unknown",
            confidence
        )

    return responses[intent], intent, confidence

# Custom CSS for a polished responsive UI
CUSTOM_CSS = """
:root {
    --primary: #6d5dfc;
    --secondary: #00b8d9;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    font-family: Inter, system-ui, sans-serif !important;
}

.hero {
    padding: 28px 30px;
    border-radius: 24px;
    margin-bottom: 18px;
    background: linear-gradient(135deg, #17153b 0%, #342a72 55%, #006f86 100%);
    color: white;
    box-shadow: 0 16px 45px rgba(20, 20, 60, .18);
}

.hero h1 {
    font-size: 34px !important;
    margin: 0 0 8px 0 !important;
}

.hero p {
    margin: 0 !important;
    opacity: .88;
    font-size: 15px;
}

.badge {
    display: inline-block;
    margin-top: 14px;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,.14);
    border: 1px solid rgba(255,255,255,.2);
    font-size: 13px;
}

.footer {
    text-align: center;
    padding: 18px;
    color: #777;
    font-size: 13px;
}

#chatbot {
    border-radius: 20px !important;
}

button {
    border-radius: 12px !important;
}
"""

def respond(message, history):
    response, intent, confidence = chatbot(message)
    return (
        f"{response}\n\n"
        f"**Detected intent:** `{intent}`  \n"
        f"**Confidence:** `{confidence * 100:.1f}%`"
    )

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div class="hero">
        <h1>🤖 Intelligent Customer Support</h1>
        <p>AI-powered assistance for orders, refunds, payments, cancellations and account support.</p>
        <div class="badge">⚡ DistilBERT + NLP Intent Classification</div>
    </div>
    """)

    chatbot_ui = gr.ChatInterface(
        fn=respond,
        title="",
        description="Ask a customer-support question below. Try one of the examples to get started.",
        examples=[
            "Where is my order?",
            "I want a refund",
            "Can I cancel my order?",
            "My product is damaged",
            "Can I pay using UPI?",
            "I forgot my password?"
        ],
        chatbot=gr.Chatbot(
            height=500,
            placeholder="Your AI support conversation will appear here..."
        ),
    )

    gr.HTML("""
    <div class="footer">
        Built as an AI/NLP project using Python, Transformers, scikit-learn and Gradio.
    </div>
    """)

if __name__ == "__main__":
   import os

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
