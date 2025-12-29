# ValtricAI Research: LoRA Rank Impact on Financial Sentiment Analysis

## Research Overview

**Research Question:** How does LoRA rank affect a small language model's ability to assess financial sentiment?

**Hypothesis:** LoRA with ~1-2% of trainable parameters can match full fine-tuning accuracy while dramatically reducing compute costs.

**Model:** distilroberta-base (82M parameters)

**Dataset:** twitter-financial-news-sentiment (3-class: bearish/bullish/neutral)

---

## Experiment Status

| Experiment | Status | Trainable Params | Accuracy | F1 Score | Time |
|------------|--------|------------------|----------|----------|------|
| Full Fine-tuning | **COMPLETED** | 82,120,707 (100%) | 88.16% | 88.20% | 1806.4s |
| LoRA Rank=4 | Pending | ~666,627 (0.81%) | - | - | - |
| LoRA Rank=64 | Pending | ~10.6M (~13%) | - | - | - |

---

## Baseline Results: Full Fine-tuning

### Configuration
- **Learning Rate:** 2e-5
- **Batch Size:** 16
- **Epochs:** 3
- **Max Sequence Length:** 128
- **Hardware:** Apple M-series (MPS)

### Performance
```
Accuracy:  88.16%
F1 Score:  88.20% (weighted)
Training Time: 1806.4 seconds (~30 minutes)
Trainable Parameters: 82,120,707 (100%)
```

### Observations
- Full fine-tuning achieves strong baseline performance on financial sentiment
- Training on MPS takes ~30 minutes (GPU acceleration will significantly reduce this)
- Model learns to distinguish bearish/bullish/neutral sentiment effectively

---

## Next Steps (Colab with GPU)

1. Run LoRA Rank=4 experiment (0.81% parameters)
2. Run LoRA Rank=64 experiment (~13% parameters)
3. Compare accuracy vs. parameter efficiency tradeoffs
4. Generate LaTeX publication materials

---

## Expected Outcomes

Based on LoRA literature, we expect:
- **LoRA r=64:** ~87-88% accuracy (within 1% of full fine-tuning)
- **LoRA r=4:** ~85-87% accuracy (slight degradation, but 99% fewer parameters)

This would demonstrate that **ValtricAI can achieve 90%+ compute savings for <2% accuracy loss**.

---

## Technical Notes

### Dataset Change
Originally planned to use `financial_phrasebank`, but HuggingFace deprecated script-based datasets. Switched to `zeroshot/twitter-financial-news-sentiment` which has:
- Similar 3-class sentiment structure
- Proper parquet format for modern datasets library
- ~9,500 samples (80/20 train/test split)

### Code Location
- Main experiment script: `lora_experiment.py`
- Colab notebook: `lora_experiment_colab.ipynb`

---

*ValtricAI Research - Last Updated: 2025-12-29*
