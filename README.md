# LoRA Rank Selection Study

Multi-seed validated study on Low-Rank Adaptation (LoRA) rank selection for financial sentiment analysis.

## Key Finding

**Diminishing returns plateau at r=16.** Accuracy gains beyond rank 16 are not statistically significant.

| Transition | Accuracy Change | p-value | Significant |
|------------|-----------------|---------|-------------|
| r=4 to r=8 | +1.35pp | 0.0044 | Yes |
| r=8 to r=16 | +0.52pp | 0.0012 | Yes |
| r=16 to r=32 | +0.35pp | 0.1372 | No |
| r=32 to r=64 | +0.14pp | 0.3469 | No |

## Results

| Configuration | Parameters | Accuracy | 95% CI |
|---------------|------------|----------|--------|
| Full Fine-Tuning | 100% | 86.85% | [85.98, 87.73] |
| LoRA r=4 | 0.81% | 82.20% | [81.17, 83.23] |
| LoRA r=8 | 0.90% | 83.55% | [82.87, 84.23] |
| LoRA r=16 | 1.08% | 84.08% | [83.32, 84.83] |
| LoRA r=32 | 1.44% | 84.42% | [83.31, 85.53] |
| LoRA r=64 | 2.16% | 84.56% | [83.62, 85.50] |

## Installation

```bash
git clone https://github.com/yourusername/lora-rank-study.git
cd lora-rank-study
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, CUDA-capable GPU recommended

## Quick Start

```bash
# Run with default settings (Twitter Financial News dataset)
python src/lora_experiment.py

# Test specific LoRA ranks
python src/lora_experiment.py --ranks 4,8,16

# Skip full fine-tuning (LoRA only)
python src/lora_experiment.py --ranks 8,16,32 --lora-only
```

## Run on Your Own Data

Use any HuggingFace dataset:

```bash
# Binary classification (e.g., IMDB sentiment)
python src/lora_experiment.py \
    --dataset imdb \
    --text-column text \
    --label-column label \
    --num-labels 2 \
    --ranks 4,8,16

# Multi-class classification
python src/lora_experiment.py \
    --dataset ag_news \
    --text-column text \
    --label-column label \
    --num-labels 4 \
    --ranks 8,16,32

# Custom model
python src/lora_experiment.py \
    --model roberta-base \
    --dataset yelp_polarity \
    --num-labels 2
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | distilroberta-base | HuggingFace model name |
| `--dataset` | zeroshot/twitter-financial-news-sentiment | HuggingFace dataset |
| `--text-column` | text | Column containing text |
| `--label-column` | label | Column containing labels |
| `--num-labels` | 3 | Number of classification labels |
| `--ranks` | 4,8,16,32,64 | Comma-separated LoRA ranks |
| `--batch-size` | 16 | Training batch size |
| `--epochs` | 3 | Number of epochs |
| `--learning-rate` | 2e-5 | Learning rate |
| `--seed` | 42 | Random seed |
| `--output-dir` | results | Output directory |
| `--lora-only` | False | Skip full fine-tuning |

## Project Structure

```
lora-rank-study/
├── src/
│   └── lora_experiment.py       # Main experiment code with CLI
├── notebooks/
│   ├── experiment_lora_rank_30runs.ipynb  # Multi-seed experiment notebook
│   └── lora_experiment_colab.ipynb
├── paper/
│   ├── lora_rank_selection_study.tex
│   ├── lora_rank_selection_study.pdf
│   ├── RESEARCH_RESULTS.md
│   └── LITERATURE_REVIEW.md
├── results/
│   └── fig1_lora_rank_multiseed_results.png
├── README.md
├── requirements.txt
├── CONTRIBUTING.md
└── LICENSE
```

## Programmatic Usage

```python
from src.lora_experiment import LoRAExperiment, ExperimentConfig

config = ExperimentConfig(
    model_name="distilroberta-base",
    dataset_name="imdb",
    num_labels=2,
    batch_size=32,
)

experiment = LoRAExperiment(config)
experiment.load_tokenizer()
experiment.load_dataset(text_column="text", label_column="label")

# Run experiments
results = experiment.run_experiments(lora_ranks=[4, 8, 16], skip_full_ft=False)

# Save results
experiment.save_results("my_results.json")
print(experiment.generate_results_table())
```

## Methodology

- **Model:** DistilRoBERTa-base (82M parameters)
- **Dataset:** Twitter Financial News Sentiment (9,543 samples)
- **Seeds:** [42, 123, 456, 789, 1337] (5 seeds x 6 configs = 30 runs)
- **LoRA Config:** target_modules=["query", "value"], alpha=2*rank, dropout=0.1
- **Statistical Tests:** Paired t-tests across seeds

## Notebooks

For the full multi-seed experiment with visualizations, use the Jupyter notebook:

```bash
jupyter notebook notebooks/experiment_lora_rank_30runs.ipynb
```

## Compiling the Paper

```bash
cd paper
pdflatex lora_rank_selection_study.tex
pdflatex lora_rank_selection_study.tex
```

## Citation

```bibtex
@techreport{hussain2025lora,
  title={Optimizing Financial Sentiment Analysis: A Systematic Study of LoRA Rank Selection},
  author={Hussain, Raja},
  institution={New York University},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
