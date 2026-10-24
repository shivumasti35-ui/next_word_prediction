
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Next Word Prediction")
st.markdown(
    "### Powered by Deep Learning (LSTM)"
)
st.write(
    "Type a sentence below and let your trained "
    "LSTM model predict the next word."
)

# -------------------------------
# File paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "lstm_model (1).h5"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
MAX_LEN_PATH = BASE_DIR / "max_len.pkl"


# -------------------------------
# Load model and files once
# -------------------------------
@st.cache_resource
def load_model_and_files():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)

    with open(MAX_LEN_PATH, "rb") as file:
        max_len = pickle.load(file)

    # Support an integer stored directly in the pickle.
    if isinstance(max_len, (list, tuple, np.ndarray)):
        max_len = int(np.asarray(max_len).reshape(-1)[0])
    else:
        max_len = int(max_len)

    return model, tokenizer, max_len


# -------------------------------
# Next-word prediction
# -------------------------------
def predict_next_word(text, model, tokenizer, max_len):
    words = text.strip().split()

    if not words:
        return None, None

    # Use the same tokenizer that was used for training.
    sequence = tokenizer.texts_to_sequences([text])

    if not sequence or not sequence[0]:
        return None, None

    # Assumption: max_len is the full training sequence length.
    input_length = max_len - 1

    padded_sequence = pad_sequences(
        sequence,
        maxlen=input_length,
        padding="pre",
        truncating="pre",
    )

    predictions = model.predict(
        padded_sequence,
        verbose=0,
    )

    probabilities = np.asarray(predictions)[0].reshape(-1)

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("The model returned invalid predictions.")

    # Convert the highest-scoring token ID into a word.
    index_word = {
        index: word
        for word, index in tokenizer.word_index.items()
    }

    predicted_id = int(np.argmax(probabilities))
    predicted_word = index_word.get(predicted_id)

    if predicted_word is None:
        return None, float(probabilities[predicted_id])

    confidence = float(probabilities[predicted_id])

    return predicted_word, confidence


# -------------------------------
# User interface
# -------------------------------
with st.sidebar:
    st.header("About this project")
    st.write("**Model:** LSTM")
    st.write("**Task:** Next-word prediction")
    st.write("**Framework:** TensorFlow / Keras")
    st.write("**Interface:** Streamlit")
    st.divider()
    st.caption(
        "Predictions depend on the text used to train the model."
    )

try:
    with st.spinner("Loading your LSTM model..."):
        model, tokenizer, max_len = load_model_and_files()

    st.success("Model loaded successfully!")

    st.caption(f"Stored maximum sequence length: {max_len}")

    user_text = st.text_area(
        "Enter your sentence",
        placeholder="Example: I love machine",
        height=120,
    )

    predict_button = st.button(
        "🔮 Predict Next Word",
        type="primary",
        use_container_width=True,
    )

    if predict_button:
        if not user_text.strip():
            st.warning("Please enter a sentence first.")
        else:
            try:
                with st.spinner("Predicting the next word..."):
                    word, confidence = predict_next_word(
                        user_text,
                        model,
                        tokenizer,
                        max_len,
                    )

                if word is None:
                    st.warning(
                        "No prediction found. Try a different sentence "
                        "using words from your training dataset."
                    )
                else:
                    st.subheader("Prediction")

                    st.markdown(
                        f"## {user_text.strip()} "
                        f"**{word}**"
                    )

                    st.metric(
                        "Model score",
                        f"{confidence * 100:.2f}%",
                    )

                    st.caption(
                        "This score is the model's output for the "
                        "predicted word. It is not necessarily a "
                        "calibrated probability."
                    )

            except Exception as error:
                st.error(f"Prediction failed: {error}")
                st.info(
                    "Check that the tokenizer, maximum sequence "
                    "length, and model match the training setup."
                )

except Exception as error:
    st.error("Could not load your model or supporting files.")
    st.code(str(error))
    st.info(
        "Verify that all three files are in the same folder "
        "as app.py and that the required packages are installed."
    )
