# Transformers from Scratch

A chapter-by-chapter implementation of Transformer neural networks in Python/TensorFlow, built up progressively from RNNs and attention mechanisms to a full sequence-to-sequence Transformer capable of English-to-German translation.

This repository is structured as a learning journey — each chapter introduces a new concept, building on the previous one until a complete Transformer model is assembled, trained, and used for real NLP tasks.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Chapter-by-Chapter Guide](#chapter-by-chapter-guide)
- [Running the Code](#running-the-code)
- [Key Concepts Covered](#key-concepts-covered)
- [Dataset](#dataset)
- [Model Architecture Summary](#model-architecture-summary)
- [Applied NLP (Chapter 23)](#applied-nlp-chapter-23)

---

## Overview

This codebase walks through the implementation of the Transformer architecture described in the seminal paper *"Attention Is All You Need"* (Vaswani et al., 2017), starting from simpler recurrent baselines and incrementally introducing every component needed for a production-grade Transformer.

The final chapters (20–22) train an encoder-decoder Transformer on a parallel English-German corpus and expose a `Translate` module for inference. Chapter 23 then demonstrates how pre-trained Transformer-based models (via HuggingFace) can be applied to summarisation and question answering.

---

## Repository Structure

```
Transformers/
│
├── appendix_02/              # Environment verification
│   ├── 01_python_versions.py     # Check core library versions
│   └── 02_deep_versions.py       # Check deep learning library versions
│
├── chapter_07/               # RNN baseline on time series
│   ├── 03_weights.py
│   ├── 04_threesteps.py
│   └── 10_simplernn.py           # SimpleRNN on monthly sunspot data
│
├── chapter_08/               # Attention fundamentals
│   └── 07_attention.py           # Scaled dot-product attention from scratch (NumPy)
│
├── chapter_09/               # RNN with attention
│   ├── 02_fibonacci.py           # Fibonacci sequence generation
│   ├── 03_split.py
│   ├── 04_rnn.py
│   ├── 05_training.py
│   ├── 07_attention.py           # Custom Keras attention layer
│   ├── 08_training.py
│   └── 09_complete.py            # Full RNN + attention on Fibonacci data
│
├── chapter_13/               # Tokenisation and encoding
│   ├── 01_encoding.py
│   ├── 02_plot.py
│   └── 03_matrix.py
│
├── chapter_14/               # Positional encoding
│   ├── 02_vectorize.py
│   ├── 03_embedding.py
│   ├── 04_positional.py
│   ├── 05_output.py
│   ├── 07_posembed.py
│   ├── 09_posencoding.py
│   ├── 11_random.py
│   └── 12_sinusoidal.py          # Learnable vs fixed sinusoidal positional encoding
│
├── chapter_15/               # Single-head self-attention (TensorFlow)
│   └── 11_testattention.py
│
├── chapter_16/               # Multi-head attention
│   └── 11_testattention.py
│
├── chapter_17/               # Transformer Encoder
│   ├── multihead_attention.py
│   ├── positional_encoding.py
│   └── 09_encoder.py             # EncoderLayer + Encoder classes
│   └── 13_testencoder.py
│
├── chapter_18/               # Transformer Decoder
│   ├── encoder.py
│   ├── multihead_attention.py
│   ├── positional_encoding.py
│   └── 05_decoder.py             # DecoderLayer + Decoder classes
│   └── 09_testdecoder.py
│
├── chapter_19/               # Full Transformer model assembly
│   ├── encoder.py
│   ├── decoder.py
│   ├── multihead_attention.py
│   ├── positional_encoding.py
│   ├── model.py                  # TransformerModel (padding + lookahead masking)
│   ├── 02_padding.py
│   ├── 04_lookahead.py
│   ├── 09_transformer.py
│   ├── 13_create_model.py
│   └── 18_encdec/                # Self-contained encoder-decoder summary
│
├── chapter_20/               # Dataset preparation + training (v1)
│   ├── prepare_dataset.py
│   ├── 01_prepare.py
│   ├── 02_testprepare.py
│   ├── model.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── multihead_attention.py
│   ├── positional_encoding.py
│   ├── 16_traintransformer.py    # Full training loop with LR scheduler + checkpointing
│   └── english-german-both.pkl   # Pre-processed parallel corpus
│
├── chapter_21/               # Training with attention weight visualisation
│   ├── (same structure as chapter_20)
│   ├── 02_model.py               # Extended model exposing attention weights
│   └── 03_plotting.py            # Attention heatmap visualisation
│
├── chapter_22/               # Inference / translation
│   ├── model.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── multihead_attention.py
│   ├── positional_encoding.py
│   ├── prepare_dataset.py
│   ├── 04_inference.py
│   └── 06_translate.py           # Translate class: greedy autoregressive decoding
│
└── chapter_23/               # Applied NLP with HuggingFace pre-trained models
    ├── article.txt               # Source article for demonstration
    ├── 02_summary.py             # Extractive summarisation with DistilBERT
    └── 03_answering.py           # Question answering with DistilBERT-SQuAD
```

---

## Prerequisites

- Python 3.8+
- TensorFlow 2.x
- NumPy, SciPy, pandas, scikit-learn, matplotlib
- HuggingFace `transformers` (chapter 23 only)
- `bert-extractive-summarizer` (chapter 23 only)

Verify your environment before starting:

```bash
python appendix_02/01_python_versions.py
python appendix_02/02_deep_versions.py
```

---

## Installation

```bash
git clone https://github.com/alketcecaj12/Transformers.git
cd Transformers

pip install tensorflow numpy scipy pandas scikit-learn matplotlib
pip install transformers bert-extractive-summarizer
```

For GPU-accelerated training (recommended for chapters 20–21):

```bash
pip install tensorflow[and-cuda]
```

---

## Chapter-by-Chapter Guide

### Appendix 02 — Environment Check

Verifies that all required libraries are installed at compatible versions. Run these scripts first.

---

### Chapter 07 — SimpleRNN Baseline

Establishes an RNN baseline for sequence prediction using TensorFlow/Keras. A `SimpleRNN` model is trained on the [monthly sunspot dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv) — a classic time series benchmark.

Key functions:
- `get_train_test()` — loads, normalises, and splits the CSV data
- `get_XY()` — builds sliding-window input/target pairs
- `create_RNN()` — constructs a `Sequential` model with a `SimpleRNN` layer
- `print_error()` / `plot_result()` — RMSE reporting and visualisation

---

### Chapter 08 — Scaled Dot-Product Attention (NumPy)

A minimal, pure-NumPy implementation of the scaled dot-product attention mechanism. Four word embeddings are defined manually; Query, Key, and Value weight matrices are randomly initialised; and attention is computed as:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

This chapter is the conceptual core of the entire repository.

---

### Chapter 09 — RNN with Custom Attention Layer

Extends the chapter 07 RNN by introducing a trainable Keras `attention` layer. The model is evaluated on a Fibonacci sequence prediction task, comparing a plain `SimpleRNN` against `RNN + Attention`, measuring improvement via MSE.

Key class: `attention(Layer)` — implements alignment scoring, softmax weighting, and context vector computation using Keras backend operations.

---

### Chapter 13 — Tokenisation and Encoding

Covers text vectorisation fundamentals — converting raw text into integer token sequences and exploring embedding matrix representations.

---

### Chapter 14 — Positional Encoding

Implements and visualises two positional encoding strategies:

**Learnable (`PositionEmbeddingLayer`):** Both word and position embeddings are randomly initialised and learned during training.

**Fixed sinusoidal (`PositionEmbeddingFixedWeights`):** Implements the encoding from *"Attention Is All You Need"* using:

```
PE(k, 2i)   = sin(k / n^(2i/d))
PE(k, 2i+1) = cos(k / n^(2i/d))
```

Both are demonstrated on two example phrases and visualised as colour matrices.

---

### Chapter 15–16 — Self-Attention and Multi-Head Attention (TensorFlow)

Progressive implementation of attention in TensorFlow:
- Chapter 15: single-head self-attention
- Chapter 16: multi-head attention with projection layers

---

### Chapter 17 — Transformer Encoder

Builds the full Transformer encoder stack:

- `AddNormalization` — residual connection + layer normalisation
- `FeedForward` — two-layer dense network with ReLU
- `EncoderLayer` — multi-head attention → dropout → add&norm → feed-forward → dropout → add&norm
- `Encoder` — stacks `n` `EncoderLayer` instances on top of positional encoding

---

### Chapter 18 — Transformer Decoder

Mirrors chapter 17 for the decoder:

- `DecoderLayer` — masked self-attention → cross-attention (attending to encoder output) → feed-forward, with add&norm and dropout at each stage
- `Decoder` — stacks `n` `DecoderLayer` instances

The decoder uses two types of masking:
1. **Lookahead mask** — prevents attending to future tokens during training
2. **Padding mask** — ignores zero-padded positions

---

### Chapter 19 — Full Transformer Model

Assembles the `TransformerModel` Keras `Model` class combining encoder and decoder, with:
- `padding_mask()` — marks zero-padded positions
- `lookahead_mask()` — generates a causal triangular mask
- A final `Dense` projection to vocabulary size

---

### Chapter 20 — Dataset Preparation and Training

Introduces `PrepareDataset` to process the `english-german-both.pkl` parallel corpus, and implements a complete training loop including:

- **Custom learning rate scheduler** (`LRScheduler`) — warmup phase followed by inverse square root decay, as per the original paper
- **Adam optimiser** with `β₁=0.9`, `β₂=0.98`, `ε=1e-9`
- **Masked loss** (`sparse_categorical_crossentropy`) ignoring padding tokens
- **Masked accuracy** metric
- **Checkpoint management** via `tf.train.CheckpointManager`
- **`@tf.function`-decorated** `train_step` for graph-mode performance

Default hyperparameters:

| Parameter | Value |
|-----------|-------|
| Attention heads (`h`) | 8 |
| Key/query dim (`d_k`) | 64 |
| Value dim (`d_v`) | 64 |
| Model dim (`d_model`) | 512 |
| Feed-forward dim (`d_ff`) | 2048 |
| Encoder/decoder layers (`n`) | 6 |
| Dropout rate | 0.1 |
| Batch size | 64 |

---

### Chapter 21 — Attention Visualisation

Extends the chapter 20 model to expose internal attention weights, enabling heatmap visualisation of what each attention head attends to during translation — a useful interpretability tool.

---

### Chapter 22 — Inference and Translation

Implements the `Translate` module for autoregressive inference:

1. Appends `<START>` and `<EOS>` tokens to the source sentence
2. Tokenises and pads the input using saved tokenisers (`enc_tokenizer.pkl`, `dec_tokenizer.pkl`)
3. Generates the target sequence token-by-token via greedy decoding
4. Stops when `<EOS>` is predicted or `dec_seq_length` is reached

Usage:

```python
from model import TransformerModel
from chapter_22.translate import Translate

inferencing_model = TransformerModel(enc_vocab_size=2404, dec_vocab_size=3864,
                                     enc_seq_length=7, dec_seq_length=12,
                                     h=8, d_k=64, d_v=64, d_model=512,
                                     d_ff=2048, n=6, rate=0)

inferencing_model.load_weights('weights/wghts16.ckpt')
translator = Translate(inferencing_model)
print(translator(['im thirsty']))
```

---

### Chapter 23 — Applied NLP with HuggingFace

Demonstrates how pre-trained Transformer-based models can be applied directly to real NLP tasks without training from scratch.

**Extractive Summarisation** (`02_summary.py`):

```python
from summarizer import Summarizer
model = Summarizer('distilbert-base-uncased')
result = model(text, num_sentences=3)
```

**Question Answering** (`03_answering.py`):

```python
from transformers import pipeline
answering = pipeline("question-answering",
                     model='distilbert-base-uncased-distilled-squad')
result = answering(question="What is BOE doing?", context=text)
```

The source article (`article.txt`) is used as context for both tasks.

---

## Running the Code

Each chapter is self-contained. Navigate to the relevant chapter directory and run the script directly:

```bash
# Chapter 7 — RNN on sunspot data
python chapter_07/10_simplernn.py

# Chapter 8 — Attention from scratch
python chapter_08/07_attention.py

# Chapter 20 — Train the Transformer
cd chapter_20
python 16_traintransformer.py

# Chapter 22 — Translate a sentence
cd chapter_22
python 06_translate.py

# Chapter 23 — Summarise an article
cd chapter_23
python 02_summary.py
python 03_answering.py
```

> **Note:** Chapters 20–22 depend on `english-german-both.pkl` being present in the chapter directory, and chapter 22 requires pre-trained weights in a `weights/` subdirectory produced by chapter 20 training.

---

## Key Concepts Covered

| Concept | Chapter(s) |
|---------|------------|
| SimpleRNN for time series | 07 |
| Scaled dot-product attention | 08 |
| Custom Keras attention layer | 09 |
| Word and positional embeddings | 13–14 |
| Sinusoidal positional encoding | 14 |
| Multi-head attention | 15–16 |
| Encoder (AddNorm, FeedForward) | 17 |
| Decoder (masked + cross-attention) | 18 |
| Padding and lookahead masks | 19 |
| Warmup learning rate scheduling | 20 |
| Masked loss and accuracy | 20 |
| Attention weight visualisation | 21 |
| Autoregressive greedy decoding | 22 |
| HuggingFace pipelines | 23 |

---

## Dataset

The parallel English-German corpus (`english-german-both.pkl`) is a pre-tokenised and pre-processed pickle file used across chapters 20–22. It contains sentence pairs used for training the translation model.

- Encoder vocabulary size: **2,404 tokens**
- Decoder vocabulary size: **3,864 tokens**
- Encoder sequence length: **7 tokens**
- Decoder sequence length: **12 tokens**

---

## Model Architecture Summary

```
Input Sentence (English)
        │
 [Positional Encoding]
        │
 ┌──────────────────┐
 │   Encoder × 6    │  ← Multi-head self-attention + FFN
 └──────────────────┘
        │
        ▼
 ┌──────────────────┐
 │   Decoder × 6    │  ← Masked self-attention + cross-attention + FFN
 └──────────────────┘
        │
   [Dense + Softmax]
        │
 Output Tokens (German)
```

The model follows the standard "Attention Is All You Need" architecture with `d_model=512`, 8 attention heads, 6 encoder and 6 decoder layers, and a feed-forward inner dimension of 2048.

---

## References

- Vaswani et al. (2017). *Attention Is All You Need.* https://arxiv.org/abs/1706.03762
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- TensorFlow documentation: https://www.tensorflow.org/api_docs
