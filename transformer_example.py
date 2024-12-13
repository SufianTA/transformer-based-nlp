# transformer-based-nlp/transformer_example.py

import torch
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Example input text
input_text = "Hello, this is a test sentence."

# Tokenize input text
inputs = tokenizer(input_text, return_tensors='pt')

# Forward pass
outputs = model(**inputs)
logits = outputs.logits
print(logits)
