
import streamlit as st
import numpy as np
import pickle
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Next Word AI",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #101827, #172554);
        color: white;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #b8c5dc;
        font-size: 16px;
        margin-bottom: 30px;
    }

    .prediction-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-label {
        color: #a5b4fc;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .prediction-word {
        color: #60a5fa;
        font-size: 36px;
        font-weight: 800;
        margin-top: 10px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        background: #2563eb;
        color: white;
        border: none;
        padding: 12px;
        font-weight: 700;
    }

    div.stButton > button:hover {
        background: #1d4ed8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL AND TOKENIZER ----------------

@st.cache_resource
def load_resources():
    model_path = Path("next_word_lstm.h5")
    tokenizer_path = Path("tokenizer.pickle")

    if not model_path.exists():
        raise FileNotFoundError("next_word_lstm.h5 not found.")

    if not tokenizer_path.exists():
        raise FileNotFoundError("tokenizer.pickle not found.")

    model = load_model(model_path)

    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)

    return model, tokenizer


# ---------------- PREDICTION FUNCTION ----------------

def predict_next_word(model, tokenizer, text, max_sequence_len):

    token_list = tokenizer.texts_to_sequences([text])[0]

    if not token_list:
        return None

    token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    prediction = model.predict(token_list, verbose=0)

    predicted_word_index = int(np.argmax(prediction, axis=1)[0])

    return tokenizer.index_word.get(predicted_word_index)


# ---------------- MAIN APPLICATION ----------------

st.markdown(
    '<div class="main-title">✨ Next Word AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">An LSTM-powered next-word prediction application</div>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.title("About")
    st.write(
        "This application uses a Stacked LSTM neural network "
        "to predict the next word based on the text you provide."
    )

    st.divider()

    st.subheader("Model Information")
    st.write("**Architecture:** Stacked LSTM")
    st.write("**Framework:** TensorFlow / Keras")
    st.write("**Task:** Next Word Prediction")

    st.divider()

    st.caption("Built with Python and Streamlit")


# Load resources
try:
    model, tokenizer = load_resources()

    max_sequence_len = model.input_shape[1] + 1

    st.success("LSTM model loaded successfully!")

    st.subheader("Enter your text")

    input_text = st.text_area(
        "Your sentence",
        value="To be or not to",
        placeholder="Start typing your sentence...",
        height=120
    )

    if st.button("✨ Predict Next Word"):

        if not input_text.strip():
            st.warning("Please enter some text first.")

        else:
            with st.spinner("Predicting the next word..."):

                next_word = predict_next_word(
                    model,
                    tokenizer,
                    input_text,
                    max_sequence_len
                )

            if next_word:
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-label">
                        Predicted Next Word
                    </div>
                    <div class="prediction-word">
                        {next_word}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.caption(
                    f"Your sentence: {input_text} **{next_word}**"
                )

            else:
                st.warning(
                    "Unable to predict a word. Try a different sentence."
                )

except Exception as error:
    st.error(f"Unable to load the application: {error}")

    st.info(
        "Make sure next_word_lstm.h5 and tokenizer.pickle "
        "are present in your project directory."
    )
