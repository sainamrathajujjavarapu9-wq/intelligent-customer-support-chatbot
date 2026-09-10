# 🤖 Intelligent Customer Support Chatbot

An AI-powered customer support chatbot built with **Python, NLP, DistilBERT, scikit-learn and Gradio**.

## Features

- Order tracking support
- Order cancellation support
- Refund support
- Damaged product support
- Payment-method support
- Password/account support
- Greeting and general-help intents
- Transformer-based text embeddings
- Intent classification using Logistic Regression
- Confidence-based fallback
- Interactive and responsive Gradio web interface
- Custom CSS styling

## Project workflow

1. Customer enters a question.
2. DistilBERT converts the question into a text embedding.
3. A Logistic Regression classifier predicts the customer intent.
4. The chatbot selects an appropriate support response.
5. A confidence score is used for fallback handling.

## Technologies

- Python
- Pandas
- NumPy
- PyTorch
- Hugging Face Transformers
- DistilBERT
- scikit-learn
- Gradio

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

The application will provide a local Gradio web interface.

## Deployment

This project can be deployed using Hugging Face Spaces with Gradio. The same repository can also be stored on GitHub for project submission and portfolio use.

## Academic note

This is a prototype customer-support system using a small demonstration dataset. It is intended for learning and project demonstration rather than production customer service.
