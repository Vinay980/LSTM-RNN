# LSTM-RNN — Next Word Prediction

<p align="center">
  <strong>A next-word prediction app powered by a stacked LSTM network and Shakespeare's Hamlet.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Task-Next%20Word%20Prediction-6f42c1" alt="Task">
</p>

## Overview

This project trains a Long Short-Term Memory (LSTM) language model on the text of Shakespeare's *Hamlet*. Given a sequence of words, the model predicts a likely next word.

A Streamlit interface makes the trained model interactive: enter a sentence or phrase and get a next-word prediction.

> **Note:** This is an educational next-word prediction experiment trained on a single literary text. It is not a general-purpose language model and its predictions are constrained by the vocabulary and patterns in *Hamlet*.

## Demo

Run the application locally to try the model:

```bash
streamlit run app.py
```

The app opens in your browser, typically at [http://localhost:8501](http://localhost:8501).

## How it works

1. **Text source** — obtain the *Hamlet* text from the NLTK Gutenberg corpus and save it as `hamlet.txt`.
2. **Tokenization** — fit a Keras `Tokenizer` on the text and map words to integer IDs.
3. **Sequence generation** — build progressively longer n-gram sequences from the text.
4. **Padding** — pad the input sequences so they have a consistent length.
5. **Model training** — train a stacked LSTM network to predict the next token.
6. **Inference** — tokenize the user's input, keep the most recent context, pad it, and select the highest-probability output token.
7. **Web app** — load the saved model and tokenizer in Streamlit to serve predictions.

## Model architecture

The LSTM model defined in the training notebook uses the following layers:

| Layer | Configuration | Role |
|---|---|---|
| Input | Sequence of token IDs | Receives the preceding words |
| Embedding | 100-dimensional vectors | Learns a dense representation for each token |
| LSTM | 150 units, `return_sequences=True` | Learns contextual patterns across the sequence |
| Dropout | 0.2 | Regularization during training |
| LSTM | 100 units | Encodes the sequence into a final representation |
| Dense | Vocabulary size, `softmax` | Produces next-token probabilities |

**Training configuration in the notebook:** categorical cross-entropy loss, Adam optimizer, accuracy metric, validation data, and early stopping.

The repository also contains GRU experiments in the notebook. The Streamlit app currently loads the saved LSTM model, `next_word_lstm.h5`.

## Repository structure

```text
LSTM-RNN/
├── app.py                                      # Streamlit inference app
├── experiemnts.ipynb                           # Data preparation, LSTM/GRU experiments
├── hamlet.txt                                  # Shakespeare's Hamlet training text
├── next_word_lstm.h5                           # Saved LSTM model
├── next_word_lstm_model_with_early_stopping.h5 # Additional saved model checkpoint
├── tokenizer.pickle                            # Fitted Keras tokenizer
└── requirements.txt                            # Python dependencies
```

## Getting started

### Prerequisites

- Python installed
- Git
- A terminal (PowerShell, Command Prompt, or a Unix shell)

### 1. Clone the repository

```bash
git clone https://github.com/Vinay980/LSTM-RNN.git
cd LSTM-RNN
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run the app

Make sure these files are in the project root:

- `app.py`
- `next_word_lstm.h5`
- `tokenizer.pickle`

Then run:

```bash
python -m streamlit run app.py
```

## Using the application

1. Open the Streamlit page in your browser.
2. Enter a phrase in the text area. The app starts with the example `To be or not to`.
3. Select **Predict Next Word**.
4. The predicted next word appears in the result card.

The prediction is selected from the model's output using the highest-probability token.

## Training notebook

Open `experiemnts.ipynb` to explore the experiments. The notebook covers:

- Loading Shakespeare's *Hamlet* through the NLTK Gutenberg corpus
- Building a tokenizer and input sequences
- Preparing padded sequences and training/validation data
- Defining and training LSTM and GRU variants
- Generating next-word predictions
- Saving the trained model and tokenizer

To regenerate the dataset from NLTK, the notebook downloads the Gutenberg corpus. The Streamlit app itself uses the saved model and tokenizer and does not need to download the corpus at runtime.

## Notes and limitations

- The model is trained on *Hamlet*, so generated words reflect the vocabulary, spelling, and style of that text.
- A next-word prediction is a statistical model output, not necessarily a grammatically or contextually correct completion.
- The app depends on the model and tokenizer files being present in the project directory.
- The repository's saved model files use the legacy HDF5 `.h5` format.

## Tech stack

- **Python** — application and training code
- **TensorFlow / Keras** — neural network and model inference
- **LSTM / GRU** — recurrent sequence-model experiments
- **NumPy** — numerical operations
- **NLTK Gutenberg** — source text corpus
- **Streamlit** — interactive web interface
- **scikit-learn** — data-splitting utilities used in the notebook

## Author

**Vinay Kumar Mahto**

- GitHub: [@Vinay980](https://github.com/Vinay980)
- Project: [LSTM-RNN](https://github.com/Vinay980/LSTM-RNN)

---

If this project helped you explore sequence modeling, feel free to ⭐ the repository.
