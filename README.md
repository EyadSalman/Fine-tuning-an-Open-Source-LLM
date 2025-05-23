# 🧠 Fine-Tuning an Open Source LLM using Lora

This project demonstrates how to fine-tune the open-source language model **TinyLlama-1.1B-Chat-v1.0** using **LoRA (Low-Rank Adaptation)**. The model is trained on a public dataset of inspirational quotes to enable it to generate motivational responses efficiently on low-resource devices.

> Example Output:  
> **"You must be very brave, though I can only be very foolish.”"I think the difference between people who are brave and those who are foolish is that the brave know when something is foolish and the foolish don't."**

---

## 🎯 Objective

- Fine-tune a small-scale open-source LLM using LoRA and 4-bit quantization
- Use the Hugging Face dataset `Abirate/english_quotes`
- Evaluate using loss, perplexity, and sample generations
- Use Hugging Face `Trainer` API for training and evaluation

---

## 📚 Dataset

- **Name:** [Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes)
- **Description:** 1000+  motivational quotes
- **Sample Format:**

- 
---

## 🧠 Model Used

- **Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Size:** ~1.1 billion parameters
- **Architecture:** Decoder-only transformer
- **Format:** Hugging Face Transformers
- **Finetuning Method:** LoRA (Parameter-Efficient Fine-Tuning)
- **Quantization:** 4-bit (`bitsandbytes` with NF4 quantization)

---

## 🔨 Training Pipeline

### ⚙️ LoRA Configuration
- Rank `r = 16`
- Alpha = 32
- Dropout = 0.1
- Target Modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`

### 🛠️ Fine-Tuning
- Load model in 4-bit using `BitsAndBytesConfig`
- Apply LoRA via `peft`
- Format and tokenize quotes to instruction-response format
- Train using Hugging Face `Trainer` API

---

## 📈 Evaluation

- **Loss:** Language modeling cross-entropy
- **Perplexity:** Computed from average training loss
- **Output Samples:** Generated using greedy decoding and sampling
- **Visualization:** Training loss plotted using `matplotlib`

---

## 📦 Installation

Ensure Python 3.8+ is installed, then run:

```bash
pip install bitsandbytes accelerate transformers datasets peft evaluate matplotlib


