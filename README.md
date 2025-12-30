# Optimizing Financial Sentiment Analysis: A Systematic Study of LoRA Rank Selection

A multi-seed validated study investigating Low-Rank Adaptation (LoRA) rank selection for financial sentiment analysis using DistilRoBERTa-base.

## Key Finding

**Diminishing returns plateau at r=16** — Accuracy gains beyond rank 16 are statistically indistinguishable from noise.

| Transition | Accuracy Change | p-value | Significant? |
|------------|-----------------|---------|--------------|
| r=4 → r=8  | +1.35pp         | 0.0044  | Yes          |
| r=8 → r=16 | +0.52pp         | 0.0012  | Yes          |
| r=16 → r=32| +0.35pp         | 0.1372  | No           |
| r=32 → r=64| +0.14pp         | 0.3469  | No           |

## Results Summary

| Configuration | Parameters | Accuracy (mean±std) | 95% CI |
|---------------|------------|---------------------|--------|
| Full Fine-Tuning | 100% | 86.85% ± 0.70% | [85.98, 87.73] |
| LoRA r=4 | 0.81% | 82.20% ± 0.83% | [81.17, 83.23] |
| LoRA r=8 | 0.90% | 83.55% ± 0.55% | [82.87, 84.23] |
| LoRA r=16 | 1.08% | 84.08% ± 0.61% | [83.32, 84.83] |
| LoRA r=32 | 1.44% | 84.42% ± 0.89% | [83.31, 85.53] |
| LoRA r=64 | 2.16% | 84.56% ± 0.76% | [83.62, 85.50] |

## Project Structure

```
.
├── lora_rank_selection_study.tex          # LaTeX paper
├── lora_rank_selection_study.pdf          # Compiled paper
├── experiment_lora_rank_30runs.ipynb      # Multi-seed experiment (5 seeds × 6 configs)
├── fig1_lora_rank_multiseed_results.png   # Results visualization (Figure 1)
├── RESEARCH_RESULTS.md                    # Detailed results summary
├── LITERATURE_REVIEW.md                   # Related work references
└── archive/                               # Old single-run files (not tracked)
```

## Methodology

- **Dataset:** Twitter Financial News Sentiment (9,543 samples, 3-class)
- **Model:** DistilRoBERTa-base (82M parameters)
- **Seeds:** [42, 123, 456, 789, 1337] (5 seeds × 6 configs = 30 runs)
- **LoRA Config:** Target modules `query`, `value`; alpha = 2×rank; dropout = 0.1
- **Statistical Tests:** Paired t-tests across seeds

## Running the Experiment

The experiment notebook requires a GPU. Run on Google Colab or a local machine with CUDA:

```bash
pip install transformers datasets peft accelerate scikit-learn matplotlib seaborn
jupyter notebook experiment_lora_rank_30runs.ipynb
```

## Compiling the Paper

```bash
pdflatex lora_rank_selection_study.tex
pdflatex lora_rank_selection_study.tex  # Run twice for references
```

## Citation

```bibtex
@techreport{hussain2025lora,
  title={Optimizing Financial Sentiment Analysis: A Systematic Study of LoRA Rank Selection},
  author={Hussain, Raja},
  institution={New York University},
  year={2025},
  note={Multi-seed validated (n=5)}
}
```

## License

MIT License
