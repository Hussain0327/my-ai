import argparse
import time
import json
import warnings
from dataclasses import dataclass, asdict
from typing import Optional, List

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")


@dataclass
class ExperimentConfig:
    model_name: str = "distilroberta-base"
    dataset_name: str = "zeroshot/twitter-financial-news-sentiment"
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    seed: int = 42
    output_dir: str = "results"
    num_labels: int = 3
    label_map: dict = None
    id_to_label: dict = None

    def __post_init__(self):
        if self.label_map is None:
            self.label_map = {f"label_{i}": i for i in range(self.num_labels)}
        if self.id_to_label is None:
            self.id_to_label = {i: f"label_{i}" for i in range(self.num_labels)}


@dataclass
class ExperimentResult:
    model: str
    lora_rank: Optional[int]
    params_updated_pct: float
    accuracy: float
    f1_score: float
    training_time_seconds: float
    training_outcome: str
    error_message: Optional[str] = None


class LoRAExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tokenizer = None
        self.dataset = None
        self.results: List[ExperimentResult] = []

    def set_seed(self, seed: int = None):
        seed = seed or self.config.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_tokenizer(self):
        print(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

    def load_dataset(self, text_column: str = "text", label_column: str = "label"):
        print(f"Loading dataset: {self.config.dataset_name}")
        dataset = load_dataset(self.config.dataset_name)

        if "train" in dataset and "test" not in dataset:
            dataset = dataset["train"].train_test_split(test_size=0.2, seed=self.config.seed)
        elif "train" in dataset and "test" in dataset:
            pass
        else:
            first_split = list(dataset.keys())[0]
            dataset = dataset[first_split].train_test_split(test_size=0.2, seed=self.config.seed)

        def tokenize_fn(examples):
            return self.tokenizer(
                examples[text_column],
                padding="max_length",
                truncation=True,
                max_length=self.config.max_length,
            )

        tokenized = dataset.map(tokenize_fn, batched=True)

        if label_column != "labels":
            tokenized = tokenized.rename_column(label_column, "labels")

        tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        print(f"  Train: {len(tokenized['train'])} | Test: {len(tokenized['test'])}")
        self.dataset = tokenized

    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        return {"accuracy": acc, "f1": f1}

    def _count_parameters(self, model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def _create_base_model(self):
        return AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            id2label=self.config.id_to_label,
            label2id=self.config.label_map,
        )

    def _create_trainer(self, model, output_dir, learning_rate):
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            seed=self.config.seed,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        return Trainer(
            model=model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

    def run_full_finetuning(self) -> ExperimentResult:
        print("\n" + "=" * 60)
        print("Full Fine-Tuning (Baseline)")
        print("=" * 60)

        self.set_seed()
        start_time = time.time()

        try:
            model = self._create_base_model()
            total_params, trainable_params = self._count_parameters(model)
            print(f"  Parameters: {total_params:,} (100% trainable)")

            output_dir = f"{self.config.output_dir}/full_ft"
            trainer = self._create_trainer(model, output_dir, self.config.learning_rate)

            print("  Training...")
            trainer.train()

            print("  Evaluating...")
            eval_results = trainer.evaluate()
            training_time = time.time() - start_time

            result = ExperimentResult(
                model=self.config.model_name,
                lora_rank=None,
                params_updated_pct=100.0,
                accuracy=eval_results["eval_accuracy"],
                f1_score=eval_results["eval_f1"],
                training_time_seconds=training_time,
                training_outcome="Success",
            )

            self._print_results(result)
            self._cleanup(model, trainer)
            return result

        except Exception as e:
            return ExperimentResult(
                model=self.config.model_name,
                lora_rank=None,
                params_updated_pct=100.0,
                accuracy=0.0,
                f1_score=0.0,
                training_time_seconds=time.time() - start_time,
                training_outcome="Fail",
                error_message=str(e),
            )

    def run_lora_finetuning(self, rank: int) -> ExperimentResult:
        print("\n" + "=" * 60)
        print(f"LoRA Fine-Tuning (rank={rank})")
        print("=" * 60)

        self.set_seed()
        start_time = time.time()

        try:
            model = self._create_base_model()

            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=rank,
                lora_alpha=rank * 2,
                lora_dropout=0.1,
                target_modules=["query", "value"],
                bias="none",
            )

            model = get_peft_model(model, lora_config)

            total_params, trainable_params = self._count_parameters(model)
            original_total = sum(
                p.numel() for n, p in model.named_parameters() if "lora" not in n.lower()
            )
            params_pct = (trainable_params / original_total) * 100
            print(f"  LoRA params: {trainable_params:,} ({params_pct:.2f}%)")

            output_dir = f"{self.config.output_dir}/lora_r{rank}"
            trainer = self._create_trainer(
                model, output_dir, self.config.learning_rate * 5
            )

            print("  Training...")
            trainer.train()

            print("  Evaluating...")
            eval_results = trainer.evaluate()
            training_time = time.time() - start_time

            result = ExperimentResult(
                model=f"{self.config.model_name} + LoRA",
                lora_rank=rank,
                params_updated_pct=round(params_pct, 2),
                accuracy=eval_results["eval_accuracy"],
                f1_score=eval_results["eval_f1"],
                training_time_seconds=training_time,
                training_outcome="Success",
            )

            self._print_results(result)
            self._cleanup(model, trainer)
            return result

        except Exception as e:
            return ExperimentResult(
                model=f"{self.config.model_name} + LoRA",
                lora_rank=rank,
                params_updated_pct=0.0,
                accuracy=0.0,
                f1_score=0.0,
                training_time_seconds=time.time() - start_time,
                training_outcome="Fail",
                error_message=str(e),
            )

    def _print_results(self, result: ExperimentResult):
        print(f"\n  Accuracy: {result.accuracy:.4f}")
        print(f"  F1 Score: {result.f1_score:.4f}")
        print(f"  Time: {result.training_time_seconds:.1f}s")

    def _cleanup(self, model, trainer):
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run_experiments(self, lora_ranks: List[int], skip_full_ft: bool = False):
        self.results = []

        if not skip_full_ft:
            self.results.append(self.run_full_finetuning())

        for rank in lora_ranks:
            self.results.append(self.run_lora_finetuning(rank))

        return self.results

    def generate_results_table(self) -> str:
        lines = [
            "\n## Results\n",
            "| Model | LoRA Rank | Params | Accuracy | F1 | Time |",
            "|-------|-----------|--------|----------|-----|------|",
        ]

        for r in self.results:
            rank_str = str(r.lora_rank) if r.lora_rank else "Full FT"
            lines.append(
                f"| {r.model} | {rank_str} | {r.params_updated_pct:.2f}% | "
                f"{r.accuracy:.4f} | {r.f1_score:.4f} | {r.training_time_seconds:.1f}s |"
            )

        return "\n".join(lines)

    def save_results(self, filepath: str = None):
        if filepath is None:
            filepath = f"{self.config.output_dir}/experiment_results.json"

        with open(filepath, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"\nResults saved to {filepath}")

    def print_summary(self):
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        successful = [r for r in self.results if r.training_outcome == "Success"]
        baseline = next((r for r in successful if r.lora_rank is None), None)

        if baseline:
            print(f"\nBaseline accuracy: {baseline.accuracy:.4f}")
            for r in successful:
                if r.lora_rank:
                    diff = (baseline.accuracy - r.accuracy) * 100
                    print(f"LoRA r={r.lora_rank}: {r.accuracy:.4f} ({diff:+.2f}pp)")
        else:
            for r in successful:
                print(f"LoRA r={r.lora_rank}: {r.accuracy:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA Rank Study: Compare full fine-tuning vs LoRA at different ranks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (Twitter Financial News dataset)
  python lora_experiment.py

  # Test specific LoRA ranks
  python lora_experiment.py --ranks 4,8,16,32,64

  # Use custom dataset from HuggingFace
  python lora_experiment.py --dataset imdb --text-column text --label-column label --num-labels 2

  # Only run LoRA experiments (skip full fine-tuning)
  python lora_experiment.py --ranks 8,16 --lora-only

  # Custom model and batch size
  python lora_experiment.py --model roberta-base --batch-size 32
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default="distilroberta-base",
        help="HuggingFace model name (default: distilroberta-base)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="zeroshot/twitter-financial-news-sentiment",
        help="HuggingFace dataset name (default: zeroshot/twitter-financial-news-sentiment)"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of text column in dataset (default: text)"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Name of label column in dataset (default: label)"
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=3,
        help="Number of classification labels (default: 3)"
    )
    parser.add_argument(
        "--ranks",
        type=str,
        default="4,8,16,32,64",
        help="Comma-separated LoRA ranks to test (default: 4,8,16,32,64)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--lora-only",
        action="store_true",
        help="Skip full fine-tuning, only run LoRA experiments"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("LoRA Rank Study")
    print("=" * 60)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nDevice: {device}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")

    lora_ranks = [int(r.strip()) for r in args.ranks.split(",")]
    print(f"LoRA ranks: {lora_ranks}")
    print(f"Skip full fine-tuning: {args.lora_only}")

    config = ExperimentConfig(
        model_name=args.model,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
        num_labels=args.num_labels,
    )

    experiment = LoRAExperiment(config)
    experiment.load_tokenizer()
    experiment.load_dataset(text_column=args.text_column, label_column=args.label_column)
    experiment.run_experiments(lora_ranks=lora_ranks, skip_full_ft=args.lora_only)

    print(experiment.generate_results_table())
    experiment.save_results()
    experiment.print_summary()


if __name__ == "__main__":
    main()
