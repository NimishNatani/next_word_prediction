import numpy as np
import pickle
import tensorflow
from tensorflow.keras.models import load_model
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('next_word_keras.keras',safe_mode=False)

with open('tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)


def predict_next_word( model, tokenizer,text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list,verbose=0)
    predicted_word_index = np.argmax(predicted,axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

st.title('Next Word Prediction App')
input_text = st.text_input("Enter the sequence of words to predict","To be or not to")
if st.button("Predict next word"):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"{input_text} {next_word}")
