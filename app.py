import pandas as pd
import re
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Fungsi untuk membersihkan teks input
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Hanya menyimpan karakter alfanumerik dan spasi
    text = text.lower()  # Mengubah teks menjadi huruf kecil
    return text

# Fungsi untuk memproses teks input menggunakan Tokenizer yang sudah dilatih
def preprocess_input(text, tokenizer, max_length=200):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    return padded_sequence

# Memuat model LSTM yang telah dilatih
model = load_model('model.h5')

# Memuat Tokenizer yang telah dilatih
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Antarmuka Streamlit
st.title('Spam Detection using LSTM')
st.write('Masukkan pesan untuk prediksi dari model LSTM.')

input_text = st.text_area('Tulis pesan di sini')

if st.button('Prediksi'):
    # Memproses teks input menggunakan Tokenizer yang sudah dilatih
    processed_input = preprocess_input(input_text, tokenizer)

    # Melakukan prediksi menggunakan model
    prediction = model.predict(processed_input)

    # Menampilkan hasil prediksi
    if prediction[0][0] >= 0.5:
        result = 'Bukan Spam'  # Pesan tidak spam
    else:
        result = 'Spam'  # Pesan spam

    percent = ((1 - (prediction[0][0])) * 100)
    st.write(f'Prediksi: {result}')
    st.write(f'Presentasi Spam: {percent:.2f}%')

