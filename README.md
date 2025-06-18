# Hierarchical Attention Network with RoBERTa for Tweet Sentiment Analysis

---

## Overview

This repository contains the implementation of a Hierarchical Attention Network (HAN) augmented with RoBERTa embeddings for binary sentiment classification of tweets. By capturing both word- and sentence-level contextual information, our HAN-RoBERTa model achieves high accuracy and offers interpretable attention visualizations.

Dataset: https://www.kaggle.com/datasets/kazanova/sentiment140

---

## Table of Contents

1. [Features](#features)
2. [Model Architecture](#model-architecture)
3. [Evaluation & Results](#evaluation--results)
4. [Attention Visualization](#attention-visualization)

---

## Features

* **RoBERTa Embeddings**: Leverages the pre‑trained RoBERTa model for rich token representations.
* **Hierarchical Structure**:

  * **Word-level Bi-GRU + Attention** captures intra-sentence semantics.
  * **Sentence-level Bi-GRU + Attention** aggregates sentence vectors into a document representation.
* **Regularization & Stability**:

  * Dropout, LayerNorm, gradient clipping, Xavier initialization.
  * Linear learning rate scheduler with warmup.
* **Interpretability**: Generates attention weight heatmaps at both word and sentence levels.

---

## Model Architecture

```text
Input Tweet → [Sentence Tokenization] → RoBERTa Token Embeddings →
    Word-level Bi-GRU → Word-level Attention → Sentence Vector →
    Sentence-level Bi-GRU → Sentence-level Attention → Document Vector →
    Fully Connected Layer + LayerNorm + Dropout → Sigmoid → Sentiment
```

Attention equations:

$$
u_{it} = \tanh(W_w h_{it} + b_w),\quad
\alpha_{it} = \frac{\exp(u_{it}^\top u_w)}{\sum_t \exp(u_{it}^\top u_w)},\quad
s_i = \sum_t \alpha_{it}\,h_{it}
$$

Where \$h\_{it}\$ is the hidden state of token \$t\$ in sentence \$i\$, and \$u\_w\$ is the trainable context vector.

---

## Evaluation & Results

* **Validation Accuracy**: \~80%
* **Validation Loss**: < 0.5
* **Generalization**: Validation accuracy slightly higher than training accuracy, indicating effective regularization.


![han_training_curves](https://github.com/user-attachments/assets/c07f481f-2870-4e10-8942-3cd203b823ea)

<small>Figure 1. Training and validation accuracy/loss over epochs.</small>

---

## Attention Visualization

The two-level attention mechanism highlights important sentences and tokens:

![attention_visualization](https://github.com/user-attachments/assets/687551b8-b138-4be8-a20a-f5a71b51a9e5)


<small>Figure 2. Sentence-level attention (left) and word-level attention (right) on a positive review sample.</small>

Example input:

> “I absolutely loved this movie! The acting was superb and the plot kept me engaged throughout. Definitely recommend it to everyone!”

Important tokens: **loved**, **recommend**, **everyone**.

---

Outputs:

* **Sentiment label** (`positive`/`negative`)
* **Attention weights** for interpretability (optional JSON).
