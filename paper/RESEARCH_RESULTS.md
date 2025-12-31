# LoRA Rank Selection Study: Results

**Status:** Multi-seed validated (n=5 per configuration)

## Abstract

Systematic investigation of LoRA rank selection for financial sentiment analysis using DistilRoBERTa-base (82M parameters) on Twitter Financial News Sentiment dataset. Key finding: diminishing returns begin at rank 16. Gains from r=16 to r=32 to r=64 are not statistically significant (p>0.1).

## Key Finding

Accuracy gains beyond rank 16 are statistically indistinguishable from noise.

| Transition | Delta Accuracy | p-value | Significant |
|------------|----------------|---------|-------------|
| r=4 to r=8 | +1.35pp | 0.0044 | Yes |
| r=8 to r=16 | +0.52pp | 0.0012 | Yes |
| r=16 to r=32 | +0.35pp | 0.1372 | No |
| r=32 to r=64 | +0.14pp | 0.3469 | No |

## Complete Results (n=5 seeds per configuration)

| Configuration | Params (%) | Accuracy (mean +/- std) | 95% CI | F1 (mean +/- std) |
|---------------|------------|-------------------------|--------|-------------------|
| Full Fine-Tuning | 100.00% | 86.85% +/- 0.70% | [85.98, 87.73] | 86.86% +/- 0.69% |
| LoRA r=4 | 0.81% | 82.20% +/- 0.83% | [81.17, 83.23] | 82.12% +/- 0.89% |
| LoRA r=8 | 0.90% | 83.55% +/- 0.55% | [82.87, 84.23] | 83.57% +/- 0.53% |
| LoRA r=16 | 1.08% | 84.08% +/- 0.61% | [83.32, 84.83] | 84.11% +/- 0.58% |
| LoRA r=32 | 1.44% | 84.42% +/- 0.89% | [83.31, 85.53] | 84.49% +/- 0.90% |
| LoRA r=64 | 2.16% | 84.56% +/- 0.76% | [83.62, 85.50] | 84.65% +/- 0.73% |

## Statistical Significance Testing

### Full FT vs LoRA (paired t-tests)

| Comparison | Delta Accuracy | t-statistic | p-value |
|------------|----------------|-------------|---------|
| Full FT vs r=4 | +4.65pp | 31.124 | <0.0001 |
| Full FT vs r=8 | +3.30pp | 18.187 | 0.0001 |
| Full FT vs r=16 | +2.78pp | 17.017 | 0.0001 |
| Full FT vs r=32 | +2.43pp | 11.234 | 0.0004 |
| Full FT vs r=64 | +2.29pp | 12.894 | 0.0002 |

### The "r=32 beats r=64" Claim

**Single-run (n=1):** r=32 (85.5%) > r=64 (85.3%) appeared significant

**Multi-seed (n=5):** r=32 (84.42%) < r=64 (84.56%) difference is noise
- Difference: -0.14pp
- p-value: 0.3469 (not significant)
- 95% CIs overlap substantially

**Conclusion:** Original finding was a statistical artifact of single-run variance.

## Run-to-Run Variance

| Configuration | Min | Max | Spread |
|---------------|-----|-----|--------|
| Full Fine-Tuning | 85.96% | 87.59% | 1.62pp |
| LoRA r=4 | 81.25% | 83.34% | 2.10pp |
| LoRA r=8 | 82.77% | 83.92% | 1.15pp |
| LoRA r=16 | 83.18% | 84.70% | 1.52pp |
| LoRA r=32 | 83.18% | 85.28% | 2.10pp |
| LoRA r=64 | 83.55% | 85.28% | 1.73pp |

Typical variance is 1.5-2.0 percentage points. Single-run differences smaller than this cannot be trusted.

## Efficiency Analysis

| Config | Accuracy Retention | Param Reduction | Significant vs Previous |
|--------|-------------------|-----------------|-------------------------|
| LoRA r=4 | 94.6% | 99.2% | - |
| LoRA r=8 | 96.2% | 99.1% | Yes (vs r=4) |
| LoRA r=16 | 96.8% | 98.9% | Yes (vs r=8) |
| LoRA r=32 | 97.2% | 98.6% | No (vs r=16) |
| LoRA r=64 | 97.4% | 97.8% | No (vs r=32) |

## Deployment Recommendations

| Use Case | Recommendation | Rationale |
|----------|----------------|-----------|
| Maximum accuracy | Full Fine-Tuning | 2.3-4.6pp better than any LoRA (p<0.001) |
| Production deployment | LoRA r=16 | Best accuracy in significant-improvement zone |
| If higher rank is free | LoRA r=32 or r=64 | Marginally better, not statistically significant |
| Memory-constrained | LoRA r=8 | Good accuracy with minimal parameters |
| Rapid experimentation | LoRA r=4 | Fastest training, acceptable for prototyping |

## Methodology

**Dataset:** Twitter Financial News Sentiment (9,543 samples)
- Train: 7,634 | Test: 1,909
- Labels: Bearish, Bullish, Neutral
- Train/test split re-randomized per seed

**Model:** DistilRoBERTa-base (82M parameters)

**Seeds:** [42, 123, 456, 789, 1337]

**LoRA Configuration:**
- Target modules: query, value
- Alpha: 2 * rank
- Dropout: 0.1

**Training:**
- Batch size: 32
- Learning rate: 2e-5 (full), 1e-4 (LoRA)
- Epochs: 3
- Precision: FP16

**Statistical Tests:** Paired t-tests (same seeds across configurations)

## Files

| File | Description |
|------|-------------|
| lora_rank_selection_study.tex | LaTeX paper |
| lora_rank_selection_study.pdf | Compiled paper |
| experiment_lora_rank_30runs.ipynb | Multi-seed experiment notebook |
| fig1_lora_rank_multiseed_results.png | Results visualization |

## Citation

```bibtex
@techreport{hussain2025lora,
  title={Optimizing Financial Sentiment Analysis: A Systematic Study of LoRA Rank Selection},
  author={Hussain, Raja},
  institution={New York University},
  year={2025}
}
```
