import streamlit as st
import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertModel
import os

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")

# Define Hierarchical BERT Model
class HierarchicalBERT(tf.keras.Model):
    def __init__(self, bert_model, lstm_units, cnn_filters, dense_units):
        super(HierarchicalBERT, self).__init__()
        self.bert = bert_model
        self.dense_sentence = tf.keras.layers.Dense(768, activation='relu')
        self.mean_pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.bilstm_encoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))
        self.conv = tf.keras.layers.Conv1D(cnn_filters, 2, activation='relu')
        self.pool = tf.keras.layers.GlobalMaxPooling1D()
        self.dense_fc = tf.keras.layers.Dense(dense_units, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        bert_output = self.bert(inputs)[0]
        sentence_encoded = self.dense_sentence(bert_output)
        context_summarized = self.mean_pooling(sentence_encoded)
        context_summarized = tf.expand_dims(context_summarized, 1)
        context_encoded = self.bilstm_encoder(context_summarized)
        context_encoded_squeezed = tf.squeeze(context_encoded, axis=1)
        context_encoded_expanded = tf.expand_dims(context_encoded_squeezed, axis=-1)
        conv_output = self.conv(context_encoded_expanded)
        pooled_output = self.pool(conv_output)
        dense_output = self.dense_fc(pooled_output)
        final_output = self.output_layer(dense_output)
        return final_output

# Load model
@st.cache_resource
def load_model():
    bert = TFBertModel.from_pretrained("bert-base-uncased")
    model = HierarchicalBERT(bert, lstm_units=128, cnn_filters=64, dense_units=32)
    model.load_weights("model/saved_model/model.h5")
    return model

# Preprocess input
def preprocess(text, tokenizer, max_length=100):
    tokenized = tokenizer(
        [text],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="np"
    )
    return tokenized["input_ids"]

# Streamlit UI
st.title("🤖 Sarcasm Detection using Hierarchical BERT")
st.write("Type a sentence and check if it's sarcastic or not!")

text_input = st.text_area("💬 Enter your comment here:", height=100)
if st.button("Predict"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        tokenizer = load_tokenizer()
        model = load_model()
        input_ids = preprocess(text_input, tokenizer)
        prediction = model(input_ids)
        result = (prediction.numpy()[0][0] > 0.5)
        if result:
            st.error("🌀 **Sarcastic**")
        else:
            st.success("✅ **Not Sarcastic**")
