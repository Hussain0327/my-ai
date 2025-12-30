# Literature Review: Parameter-Efficient Fine-Tuning for Financial NLP

A curated collection of research papers supporting the LoRA efficiency study on financial sentiment analysis.

---

## 1. Core LoRA Papers

### 1.1 LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)

**The foundational paper.**

| Field | Value |
|-------|-------|
| Authors | Edward Hu, Yelong Shen, Phillip Wallis, et al. (Microsoft) |
| Published | June 2021, ICLR 2022 |
| PDF | [arxiv.org/pdf/2106.09685](https://arxiv.org/pdf/2106.09685) |

**Key Findings:**
- Pre-trained models have low "intrinsic rank" — weight updates during fine-tuning don't need full dimensionality
- LoRA reduces trainable parameters by **10,000x** and GPU memory by **3x** compared to full fine-tuning
- Achieves **on-par or better performance** than full fine-tuning on RoBERTa, DeBERTa, GPT-2, GPT-3
- **No inference latency** — LoRA weights merge into base model at deployment
- Storage: 100 adapted models = 354GB vs 35TB for full fine-tuned models

**Technical Details:**
- Injects trainable rank decomposition matrices into transformer layers
- Typically applied to query (Wq) and value (Wv) projection matrices
- Rank r controls capacity vs efficiency tradeoff

**Citation:**
```bibtex
@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward and others},
  booktitle={ICLR},
  year={2022}
}
```

---

### 1.2 DLoRA: Distributed Parameter-Efficient Fine-Tuning (Gao & Zhang, 2024)

**Privacy-preserving distributed LoRA.**

| Field | Value |
|-------|-------|
| Authors | Chao Gao, Sai Qian Zhang |
| Published | EMNLP 2024 Findings |
| PDF | [aclanthology.org/2024.findings-emnlp.802.pdf](https://aclanthology.org/2024.findings-emnlp.802.pdf) |

**Key Findings:**
- Distributes fine-tuning between cloud (frozen backbone) and user devices (LoRA modules)
- **Kill and Revive (KR) algorithm** identifies "active" vs "idle" parameter modules during training
- Reduces computation and communication workload by **>80%**
- Maintains user data privacy by keeping sensitive data local

**Relevance to Our Study:**
- Shows that not all LoRA parameters are equally important
- Justifies investigating different rank configurations
- Demonstrates LoRA's practical deployment advantages

**Citation:**
```bibtex
@inproceedings{gao2024dlora,
  title={DLoRA: Distributed Parameter-Efficient Fine-Tuning Solution for Large Language Model},
  author={Gao, Chao and Zhang, Sai Qian},
  booktitle={Findings of EMNLP},
  year={2024}
}
```

---

### 1.3 LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4 (Zhao et al., 2024)

**Large-scale LoRA benchmarking study.**

| Field | Value |
|-------|-------|
| Authors | Justin Zhao, Timothy Wang, et al. (Predibase) |
| Published | May 2024 |
| PDF | [arxiv.org/pdf/2405.00732](https://arxiv.org/pdf/2405.00732) |

**Key Findings:**
- Evaluated **310 models** across 10 base models and 31 tasks
- 4-bit LoRA fine-tuned models outperform base models by **34 points** and GPT-4 by **10 points** on average
- **224 out of 310 models** exceeded GPT-4 benchmark
- Mistral-7B and Zephyr-7b-beta emerge as leaders
- Fine-tuned 7B models beat 2B models on 29/31 tasks
- Each model fine-tuned for **$8** on Predibase
- 25 LoRA models served on **single A100 GPU** via LoRAX

**Relevance to Our Study:**
- Validates that LoRA can produce competitive models
- Shows cost-effectiveness of LoRA approach
- Demonstrates multi-adapter serving capabilities

**Citation:**
```bibtex
@article{zhao2024loraland,
  title={LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4},
  author={Zhao, Justin and others},
  journal={arXiv preprint arXiv:2405.00732},
  year={2024}
}
```

---

## 2. Financial NLP Papers

### 2.1 FinBERT: Financial Sentiment Analysis with BERT (Araci, 2019)

**The benchmark for financial sentiment.**

| Field | Value |
|-------|-------|
| Authors | Dogu Araci (Prosus AI) |
| Published | 2019 |
| GitHub | [github.com/ProsusAI/finBERT](https://github.com/ProsusAI/finBERT) |

**Key Findings:**
- Achieved **97% accuracy** on Financial PhraseBank (full agreement subset)
- 6 percentage points higher than previous SOTA (FinSSLX)
- 15 percentage point improvement over general BERT
- Demonstrates importance of domain-specific pre-training

**Benchmark Reference:**
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| FinBERT | 87-90% | 0.85-0.91 |
| Vanilla BERT | 96% (agreement subset) | - |
| FinancialBERT | 99% | 0.98 |

---

### 2.2 Financial PhraseBank Dataset (Malo et al., 2014)

**Standard benchmark dataset for financial sentiment.**

| Field | Value |
|-------|-------|
| Source | LexisNexis financial news |
| Size | 4,845 sentences |
| Labels | Positive, Negative, Neutral |
| Annotators | 16 finance/business experts |

**Dataset Characteristics:**
- Sentences selected from financial news
- Annotated based on expected stock price impact
- Multiple agreement levels available (50%, 66%, 75%, 100%)
- Standard train/test splits for reproducibility

**Why We Use Twitter Financial News Instead:**
- HuggingFace deprecated script-based datasets
- Twitter dataset has similar 3-class structure
- ~9,500 samples with proper parquet format
- More modern/accessible for reproduction

---

## 3. Cutting-Edge Research (2024-2025)

### 3.1 Evolutionary Optimization of Model Merging Recipes (Sakana AI, 2025)

**Automated model optimization.**

| Field | Value |
|-------|-------|
| Authors | Sakana AI Team |
| Published | Nature Machine Intelligence, January 2025 |
| Link | [sakana.ai/evolutionary-model-merge](https://sakana.ai/evolutionary-model-merge/) |

**Key Findings:**
- Uses evolutionary algorithms to discover optimal model merging strategies
- Operates in both **parameter space** and **data flow space**
- Creates 100+ model offspring per generation, benchmarks, selects best
- Japanese Math LLM achieved **SOTA** on Japanese benchmarks
- Cross-domain merging: combines models from different domains (Math + Language)

**Relevance to Our Study:**
- Future work: evolutionary search for optimal LoRA rank
- Demonstrates automated hyperparameter optimization potential
- Shows that model efficiency can be systematically optimized

**Citation:**
```bibtex
@article{sakana2025evolutionary,
  title={Evolutionary Optimization of Model Merging Recipes},
  author={Sakana AI},
  journal={Nature Machine Intelligence},
  year={2025}
}
```

---

### 3.2 The AI Scientist (Sakana AI & Oxford, 2024)

**Automated scientific discovery.**

| Field | Value |
|-------|-------|
| Authors | Sakana AI, Oxford |
| Published | August 2024 |
| PDF | [arxiv.org/pdf/2408.06292](https://arxiv.org/pdf/2408.06292) |

**Relevance:**
- Demonstrates trend toward automated ML research
- Our study: manual exploration of rank-accuracy tradeoff
- Future: automated discovery of optimal configurations

---

## 4. Key Metrics & Benchmarks

### Financial Sentiment Accuracy Targets

| Model Type | Expected Accuracy | Notes |
|------------|-------------------|-------|
| Random Baseline | 33% | 3-class classification |
| General BERT | 70-80% | No domain training |
| DistilRoBERTa (Full FT) | 85-88% | Our baseline |
| FinBERT | 87-90% | Domain-specific |
| FinancialBERT | 97-99% | SOTA (full agreement) |

### LoRA Efficiency Expectations

| Configuration | Parameters | Expected Accuracy | Memory |
|---------------|------------|-------------------|--------|
| Full Fine-Tuning | 100% | 87-88% | 4-6 GB |
| LoRA r=64 | ~2% | 85-87% | ~2 GB |
| LoRA r=16 | ~0.5% | 83-86% | ~1.5 GB |
| LoRA r=4 | ~0.1% | 80-84% | ~1.2 GB |

---

## 5. Citation Templates

### For Introduction
> "As demonstrated by Hu et al. (2021) and extended by recent distributed approaches (Gao & Zhang, 2024), parameter-efficient fine-tuning enables practical deployment of large language models in resource-constrained environments."

### For Methodology
> "Following the efficiency protocols validated in LoRA Land (Zhao et al., 2024), we investigate the trade-off between model sparsity (rank) and financial sentiment accuracy using the approach established by FinBERT (Araci, 2019)."

### For Future Work
> "Building on Sakana AI's evolutionary optimization framework (2025), future work could automatically discover optimal LoRA configurations for financial NLP tasks."

---

## 6. Paper Download Links

| Paper | Direct PDF |
|-------|-----------|
| LoRA (Hu et al., 2021) | [arxiv.org/pdf/2106.09685](https://arxiv.org/pdf/2106.09685) |
| DLoRA (EMNLP 2024) | [aclanthology.org/2024.findings-emnlp.802.pdf](https://aclanthology.org/2024.findings-emnlp.802.pdf) |
| LoRA Land (2024) | [arxiv.org/pdf/2405.00732](https://arxiv.org/pdf/2405.00732) |
| Sakana Evolutionary | [arxiv.org/pdf/2403.13187](https://arxiv.org/pdf/2403.13187) |
| AI Scientist | [arxiv.org/pdf/2408.06292](https://arxiv.org/pdf/2408.06292) |
| FinEAS | [arxiv.org/pdf/2111.00526](https://arxiv.org/pdf/2111.00526) |

---

*Compiled December 2025*
