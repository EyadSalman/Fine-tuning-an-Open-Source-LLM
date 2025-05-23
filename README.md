# 🧠 Fine-Tuning an Open Source LLM using LoRA

This project presents a complete pipeline to fine-tune a compact but powerful open-source language model — **TinyLlama-1.1B-Chat-v1.0** — using **LoRA (Low-Rank Adaptation)**. We train it on a public dataset of inspirational quotes to generate human-like motivational responses with minimal compute.

> Example Output:  
> **"You must be very brave, though I can only be very foolish.”"I think the difference between people who are brave and those who are foolish is that the brave know when something is foolish and the foolish don't."**

---

## 🎯 Objective

- Fine-tune a small-scale LLM with low compute using 4-bit quantization and LoRA
- Train on a simple instruction-style dataset (`Abirate/english_quotes`)
- Evaluate model performance using loss, perplexity, and qualitative generation
- Demonstrate instruction-tuned behavior through inference prompts

---

## 📚 Dataset

- [Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes)  
- 1000+ motivational quotes


---

## 🧠 Model & Training Setup

- **Base Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Parameters:** 1.1B
- **Quantization:** 4-bit (NF4, with `bitsandbytes`)
- **Fine-tuning method:** LoRA via `peft`

### LoRA Config
- `r = 16`
- `lora_alpha = 32`
- `dropout = 0.1`
- Target Modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`

---

## 📂 Project Structure
.
├── FinetuningLLM.ipynb      # Full training notebook
├── outputs/                 # Saved model checkpoints
├── README.md                # Project documentation
└── requirements.txt         # Library dependencies


