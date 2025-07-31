
#  Sarcasm Detection using Hierarchical BERT + BiLSTM + CNN

This project implements a sarcasm detection system using a hybrid NLP architecture combining **BERT**, **BiLSTM**, and **CNN layers**. It identifies whether a given news headline is sarcastic or not.

## 📁 Dataset

- Dataset: [`Sarcasm_Headlines_Dataset.json`](https://www.kaggle.com/danofer/sarcasm)

## 🧪 Model Architecture

- **BERT Tokenizer** for contextual token embeddings
- **TFBertModel** from Hugging Face as base
- **BiLSTM** layer to capture sequence dependencies
- **CNN** layers to extract n-gram level features
- **Dense layer** for binary classification

## 📊 Performance

- Trained on 10,000 samples
- Robust at detecting sarcastic patterns

## 🛠️ Tech Stack

- Python
- TensorFlow + Keras
- Hugging Face Transformers
- Pandas, NumPy
- Streamlit (for deployment)

