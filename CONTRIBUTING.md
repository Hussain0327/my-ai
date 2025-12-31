# Contributing

Contributions are welcome. Here's how to help.

## Ways to Contribute

1. **Test on new datasets** - Run experiments on different HuggingFace datasets and share results
2. **Test on new models** - Try different base models (BERT, RoBERTa, etc.)
3. **Add PEFT methods** - Implement comparisons with other methods (adapters, prefix tuning)
4. **Improve documentation** - Fix typos, add examples, clarify instructions
5. **Bug fixes** - Report or fix issues

## Setup

```bash
git clone https://github.com/yourusername/lora-rank-study.git
cd lora-rank-study
pip install -r requirements.txt
```

## Code Style

- Use clear variable names
- Keep functions focused and small
- No excessive comments - code should be self-explanatory
- Follow existing patterns in the codebase

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Test your changes
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

## Reporting Issues

When reporting bugs, include:
- Python version
- PyTorch version
- GPU (if applicable)
- Steps to reproduce
- Error message

## Adding New Experiments

If you run experiments on new datasets/models:

1. Use the same methodology (multiple seeds, statistical tests)
2. Document your configuration
3. Include results in a clear format
4. Add to `paper/RESEARCH_RESULTS.md` if significant

## Questions

Open an issue for questions about the codebase or methodology.
