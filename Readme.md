<p align="center">
  <img src="https://img.shields.io/badge/Qwen1.5--0.5B-Fine--Tuned-blueviolet?style=for-the-badge&logo=huggingface" />
  <img src="https://img.shields.io/badge/LoRA-PEFT-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-yellow?style=for-the-badge&logo=googlecolab" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

# 🛒 Fine-Tuning Qwen1.5-0.5B-Chat for Amazon Product Content Generation

> **Generate realistic product names and descriptions from just a category label** — powered by a LoRA fine-tuned Qwen1.5-0.5B-Chat model trained on real Amazon product data.

---

## 📑 Table of Contents

- [🤖 Overview](#-overview)
- [✨ Features](#-features)
- [🧰 Tech Stack](#-tech-stack)
- [🧠 Architecture](#-architecture)
- [🔁 Training Pipeline Flow](#-training-pipeline-flow)
- [📚 Dataset & Data Processing](#-dataset--data-processing)
- [🎥 Demo](#-demo)
- [🚀 Getting Started](#-getting-started)
  - [✅ Prerequisites](#-prerequisites)
  - [📦 Installation](#-installation)
  - [▶️ Run the App](#️-run-the-app)
- [⚙️ Configuration](#️-configuration)
- [🔎 How It Works (Step-by-Step)](#-how-it-works-step-by-step)
- [🛠️ Customization](#️-customization)
- [🧯 Troubleshooting](#-troubleshooting)
- [⚠️ Known Limitations](#️-known-limitations)
- [🔐 Security Notes](#-security-notes)
- [🗺️ Roadmap Ideas](#️-roadmap-ideas)
- [🙏 Acknowledgements / Sources](#-acknowledgements--sources)
- [📄 License](#-license)
- [📁 Project Structure](#-project-structure)

---

## 🤖 Overview

This project fine-tunes **Qwen1.5-0.5B-Chat** — a lightweight yet capable causal language model — using **LoRA (Low-Rank Adaptation)** on a real-world Amazon product dataset. Given only a **product category** (e.g., `Smartphones`, `BatteryChargers`), the model learns to generate:

| Task Type              | Example Input        | Example Output                                          |
|------------------------|----------------------|---------------------------------------------------------|
| **Product Name**       | `Smartphones`        | `Samsung Galaxy M14 5G (Berry Blue, 6GB, 128GB)`        |
| **Product Description**| `WirelessEarbuds`    | `Immersive sound with deep bass, 24h battery life...`   |

### Why This Matters

- **E-commerce Automation**: Auto-generate catalog content for thousands of products.
- **Content at Scale**: Marketing teams can bootstrap product listings from category labels alone.
- **Efficient Fine-Tuning**: LoRA trains only **0.74%** of total parameters — enabling fine-tuning on free-tier Google Colab GPUs.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔧 **LoRA Fine-Tuning** | Parameter-efficient fine-tuning — only 1.8M trainable params out of 464M |
| 📊 **Comprehensive Evaluation** | ROUGE-1/2/L, BLEU-1/2/3/4, METEOR, BERTScore, Perplexity |
| 🧪 **Dual Notebook Workflow** | Separate notebooks for training and evaluation — clean separation of concerns |
| 📈 **Training Visualization** | Real-time loss curves for both training and evaluation |
| 🔀 **Dual Task Support** | Single model handles both Product Name and Product Description generation |
| 💾 **Model Merging** | LoRA adapters merged back into base model for standalone deployment |
| 🆓 **Free-Tier Friendly** | Designed to run on Google Colab free tier (T4 GPU) |
| 📋 **Gap Analysis** | Automatic train vs. test performance comparison to detect overfitting |

---

## 🧰 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Base Model** | [Qwen1.5-0.5B-Chat](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat) | Pre-trained causal language model |
| **Fine-Tuning** | [PEFT (LoRA)](https://github.com/huggingface/peft) | Parameter-efficient adapter training |
| **Training** | [HuggingFace Transformers](https://github.com/huggingface/transformers) | Trainer API, tokenization, generation |
| **Optimization** | [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) | 8-bit paged AdamW optimizer |
| **Acceleration** | [HuggingFace Accelerate](https://github.com/huggingface/accelerate) | Mixed-precision and distributed setup |
| **Data** | [HuggingFace Datasets](https://github.com/huggingface/datasets) | Dataset loading, splitting, mapping |
| **Evaluation** | `rouge-score`, `nltk`, `bert-score` | Multi-metric generation quality assessment |
| **Environment** | Google Colab + Google Drive | Free GPU compute + persistent storage |
| **Language** | Python 3.10+ | Core programming language |

---

## 🧠 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE OVERVIEW                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│   │  Amazon CSV   │───▶│  Data Processor  │───▶│  HF Dataset  │  │
│   │  (Raw Data)   │    │  (Clean/Format)  │    │ (Train/Test) │  │
│   └──────────────┘    └──────────────────┘    └──────┬───────┘  │
│                                                       │          │
│                                                       ▼          │
│   ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│   │  Qwen1.5     │───▶│   LoRA Adapter   │───▶│   Trainer    │  │
│   │  0.5B-Chat   │    │  (r=8, α=16)     │    │  (500 steps) │  │
│   │  (Frozen)    │    │  q/k/v/o_proj     │    │              │  │
│   └──────────────┘    └──────────────────┘    └──────┬───────┘  │
│                                                       │          │
│                                                       ▼          │
│   ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│   │  Merged      │◀──│   Merge Weights   │◀──│  LoRA Saved  │  │
│   │  Model       │    │  (merge_and_     │    │  Adapter     │  │
│   │  (Deploy)    │    │   unload)         │    │  Checkpoint  │  │
│   └──────┬───────┘    └──────────────────┘    └──────────────┘  │
│          │                                                       │
│          ▼                                                       │
│   ┌──────────────────────────────────────┐                      │
│   │         EVALUATION SUITE             │                      │
│   │  ROUGE │ BLEU │ METEOR │ BERTScore  │                      │
│   │              Perplexity              │                      │
│   └──────────────────────────────────────┘                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### LoRA Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `r` | 8 | Rank of the low-rank decomposition |
| `lora_alpha` | 16 | Scaling factor (effective LR = α/r = 2) |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Attention projection layers |
| `lora_dropout` | 0.05 | Dropout on LoRA layers |
| `bias` | `none` | No bias terms trained |
| `task_type` | `CAUSAL_LM` | Causal language modeling objective |

> **Trainable Parameters**: 1,769,472 / 463,583,232 ≈ **0.38%** of total model

---

## 🔁 Training Pipeline Flow

```
   ┌─────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
   │  Load    │────▶│  Prepare │────▶│  Apply   │────▶│  Train   │
   │  CSV     │     │  Dataset │     │  LoRA    │     │  Model   │
   └─────────┘     └──────────┘     └──────────┘     └────┬─────┘
                                                           │
        ┌──────────────────────────────────────────────────┘
        │
        ▼
   ┌─────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
   │  Save   │────▶│  Merge   │────▶│  Test    │────▶│ Evaluate │
   │  LoRA   │     │  Weights │     │  Model   │     │ Metrics  │
   └─────────┘     └──────────┘     └──────────┘     └──────────┘
```

### Detailed Steps

| Step | Notebook | Action |
|------|----------|--------|
| 1 | `Fine_Tuning_Qwen.ipynb` | Load & preprocess Amazon product CSV |
| 2 | `Fine_Tuning_Qwen.ipynb` | Create prompt-formatted HuggingFace Dataset |
| 3 | `Fine_Tuning_Qwen.ipynb` | Load Qwen1.5-0.5B-Chat + tokenizer |
| 4 | `Fine_Tuning_Qwen.ipynb` | Attach LoRA adapters to attention layers |
| 5 | `Fine_Tuning_Qwen.ipynb` | Train for 500 steps (eval every 25 steps) |
| 6 | `Fine_Tuning_Qwen.ipynb` | Save LoRA adapter + merge into base model |
| 7 | `Fine_Tuning_Qwen.ipynb` | Quick ROUGE evaluation + sample generations |
| 8 | `Test_Finetuned_Model.ipynb` | Load merged model from Drive |
| 9 | `Test_Finetuned_Model.ipynb` | Generate predictions on 20 test + 10 train samples |
| 10 | `Test_Finetuned_Model.ipynb` | Compute ROUGE, BLEU, METEOR, BERTScore, Perplexity |
| 11 | `Test_Finetuned_Model.ipynb` | Generate summary report with gap analysis |

---

## 📚 Dataset & Data Processing

### Source Data

| Property | Details |
|----------|---------|
| **File** | `amazon_product_details.csv` |
| **Source** | Amazon product listings |
| **Key Columns** | `category`, `product_name`, `about_product` |
| **Split** | 75% Train / 25% Test (shuffled, seed=0) |

### Data Transformation

The raw CSV is transformed into two task-oriented datasets that are then combined:

```
Raw CSV Record:
  category: "Electronics|Mobiles|Smartphones"
  product_name: "Samsung Galaxy M14 5G..."
  about_product: "Immersive 16.72cm display..."

        ↓  Split into TWO records  ↓

Record 1 (Product Name Task):
  category: "Smartphones"           ← Last segment of pipe-delimited category
  task_type: "Product Name"
  text: "Samsung Galaxy M14 5G..."

Record 2 (Product Description Task):
  category: "Smartphones"
  task_type: "Product Description"
  text: "Immersive 16.72cm display..."
```

### Prompt Template

```
Given the product category, you need to generate a '{task_type}'.
### Category: {category}
### {task_type}: {text}
```

### Tokenization

| Parameter | Value |
|-----------|-------|
| `max_length` | 400 tokens |
| `padding` | `max_length` (left-padded) |
| `truncation` | `True` |
| `EOS token` | Appended to all inputs |

---

## 🎥 Demo

### Training Loss Curves

The training process produces both training and evaluation loss curves logged every 25 steps:

```
Step  25  │ Train Loss: ~3.2  │ Eval Loss: ~3.0
Step 100  │ Train Loss: ~2.5  │ Eval Loss: ~2.4
Step 250  │ Train Loss: ~1.8  │ Eval Loss: ~1.9
Step 500  │ Train Loss: ~1.5  │ Eval Loss: ~1.6
```

### Adjust LoRA for More/Less Capacity

```python
# More capacity (slower training, better results)
config = LoraConfig(r=16, lora_alpha=32, ...)

# Less capacity (faster training, possibly worse results)
config = LoraConfig(r=4, lora_alpha=8, ...)

# Target more layers
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### Train Longer or Shorter

```python
# In TrainingArguments
max_steps=1000,         # Longer training
learning_rate=1e-5,     # Lower LR for longer training
save_steps=50,          # Less frequent saves
```

### Add More Task Types

```python
# Example: Add "Product Tagline" as a new task
tagline = df[['category', 'tagline']].rename(columns={'tagline': 'text'})
tagline['task_type'] = 'Product Tagline'
df = pd.concat([products, description, tagline], ignore_index=True)
```

---

## 🧯 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `OutOfMemoryError` | GPU VRAM exhausted | Reduce `per_device_train_batch_size` to 1, or use `gradient_accumulation_steps=4` |
| `CUDA out of memory` during eval | Large generation batch | Reduce `max_new_tokens` or evaluate fewer samples |
| Model outputs gibberish | Undertrained | Increase `max_steps` to 1000+ or lower `learning_rate` |
| Model repeats itself | Repetition penalty too low | Increase `repetition_penalty` to 1.3+ |
| `FileNotFoundError` for model | Wrong Drive path | Verify paths with `os.listdir()` — check notebook path cells |
| `tokenizer.pad_token is None` | Decoder-only LM quirk | Already handled: `tokenizer.pad_token = tokenizer.eos_token` |
| Very high perplexity (>1000) | Degenerate outputs | Filter outliers (already handled in code: `if ppl < 10000`) |
| Training loss doesn't decrease | Learning rate too low/high | Try `learning_rate` in range `[1e-5, 5e-5]` |
| Google Drive disconnects | Colab timeout | Use `Colab Pro` or save checkpoints frequently (`save_steps=25`) |
| `ImportError` for packages | Package not installed | Run the `pip install` cells at the top first |

---

## ⚠️ Known Limitations

| Limitation | Details |
|------------|---------|
| **Small Base Model** | Qwen1.5-0.5B is a compact model — generation quality is limited compared to 7B+ models |
| **Low ROUGE/BLEU** | The model paraphrases rather than copying reference text — expected for generative models |
| **English Only** | Training data and evaluation are English-only |
| **Category Dependency** | Performance varies by category — categories with more training samples produce better results |
| **No Hallucination Control** | The model may generate plausible-sounding but factually incorrect product details |
| **Single Dataset** | Trained only on Amazon product data — may not generalize to other e-commerce platforms |
| **No Quantization at Inference** | Merged model runs in FP16 — could be further optimized with GPTQ/AWQ |
| **Colab Dependency** | Designed for Google Colab — running locally requires path adjustments |
| **Token Limit** | Max 400 tokens — very long product descriptions may be truncated |

---

## 🔐 Security Notes

| ⚠️ Security Consideration | Recommendation |
|---------------------------|----------------|
| **Google Drive Paths** | Hardcoded paths are exposed in notebooks — avoid committing notebooks with sensitive paths |
| **API Keys** | No API keys are used in this project — all models are loaded locally |
| **Model Outputs** | Generated content should be reviewed before publishing — models can produce misleading text |
| **Dataset Privacy** | The Amazon product dataset is publicly available — but verify licensing before commercial use |
| **Model Weights** | If sharing the fine-tuned model, ensure compliance with Qwen's model license (Apache 2.0) |
| **Colab Sessions** | Colab sessions may leave model weights in temporary storage — clear `/content/` after use |

---

## 🗺️ Roadmap Ideas

- [ ] 🔄 **Quantize merged model** with GPTQ/AWQ for faster inference
- [ ] 🌐 **Build a Gradio/Streamlit UI** for interactive product content generation
- [ ] 📊 **Scale training data** with more Amazon categories and products
- [ ] 🧪 **Experiment with larger models** (Qwen1.5-1.8B, Qwen2-7B)
- [ ] 🏷️ **Add more task types**: product taglines, bullet points, SEO keywords
- [ ] 🔁 **Implement DPO/RLHF** for preference-aligned generation
- [ ] 📈 **Add W&B / MLflow logging** for experiment tracking
- [ ] 🐳 **Dockerize** the inference pipeline for deployment
- [ ] 🌍 **Multi-language support** — fine-tune on multilingual product data
- [ ] 📦 **Push to HuggingFace Hub** — share the fine-tuned adapter publicly
- [ ] ⚡ **vLLM / TGI integration** for production-grade serving
- [ ] 🧪 **A/B testing framework** to compare generations from different checkpoints

---

## 🙏 Acknowledgements / Sources

| Resource | Credit |
|----------|--------|
| **Qwen1.5-0.5B-Chat** | [Alibaba Cloud / Qwen Team](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat) |
| **LoRA Paper** | [Hu et al., 2021 — "LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685) |
| **PEFT Library** | [HuggingFace PEFT](https://github.com/huggingface/peft) |
| **HuggingFace Transformers** | [HuggingFace](https://github.com/huggingface/transformers) |
| **BERTScore** | [Zhang et al., 2020](https://arxiv.org/abs/1904.09675) |
| **Amazon Product Dataset** | Publicly available Amazon product listings |
| **Google Colab** | Free GPU compute platform by Google |
| **BitsAndBytes** | [Tim Dettmers](https://github.com/TimDettmers/bitsandbytes) |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

The base model (Qwen1.5-0.5B-Chat) is licensed under **Apache 2.0** by Alibaba Cloud.

---

## 📁 Project Structure

```
Fine-Tuning-Qwen/
│
├── 📓 Fine_Tuning_Qwen.ipynb          # Main training notebook
│   ├── Data loading & preprocessing
│   ├── Model & tokenizer setup
│   ├── LoRA configuration & attachment
│   ├── Training (500 steps)
│   ├── Loss visualization
│   ├── LoRA adapter saving
│   ├── Model merging (LoRA → base)
│   ├── Quick ROUGE evaluation
│   └── Sample generation tests
│
├── 📓 Test_Finetuned_Model.ipynb       # Comprehensive evaluation notebook
│   ├── Merged model loading
│   ├── Dataset reconstruction
│   ├── Batch prediction generation
│   ├── ROUGE-1/2/L calculation
│   ├── BLEU-1/2/3/4 calculation
│   ├── METEOR calculation
│   ├── BERTScore calculation
│   ├── Perplexity calculation
│   ├── Metrics summary table
│   └── Gap analysis (train vs test)
│
├── 📊 amazon_product_details.csv       # Source dataset
│
├── 📄 README.md                        # This file
│
├── 📄 LICENSE                          # MIT License
│
└── 📁 (Generated on Google Drive)
    ├── 📁 train-dir/                   # Training checkpoints
    │   ├── checkpoint-25/
    │   ├── checkpoint-50/
    │   ├── ...
    │   └── logs/                       # TensorBoard logs
    │
    ├── 📁 qwen-lora-adapter/          # Saved LoRA adapter weights
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   ├── tokenizer.json
    │   └── ...
    │
    └── 📁 qwen-merged/                # Final merged model (ready for deployment)
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── ...
```

---

<p align="center">
  <b>⭐ If this project helped you, consider giving it a star! ⭐</b>
</p>

<p align="center">
  Made with ❤️ using HuggingFace 🤗 + LoRA + Qwen
</p>
