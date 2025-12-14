# LLM Judge Stress Testing

Research codebase for testing LLM judgments under systematic perturbations (authority framing, cultural bias, verbosity, etc.).

## Installation

```bash
pip install -e .

# With optional backends
pip install -e .[anthropic]    # Anthropic Claude
pip install -e .[openai]       # OpenAI models
pip install -e .[hf]           # HuggingFace local models
pip install -e .[all]          # All backends
```

## Quick Start

```bash
# Run with mock backend (no API keys needed)
python -m src.cli run --config configs/exp_toy_mock.yaml

# Run with Claude
python -m src.cli run --config configs/exp_claude.yaml

# Run with multiple backends
python -m src.cli run --config configs/exp_mix_all.yaml
```

## Configuration

All experiments are configured via YAML files in `configs/`. Key features:

- **Backend swapping**: Change models/backends via config only
- **Deterministic caching**: Avoids re-billing APIs on reruns
- **Mixed backends**: Run multiple models in parallel
- **Systematic perturbations**: Authority, cultural framing, verbosity, controls

## Structure

- `src/`: Core implementation
- `configs/`: Experiment configurations
- `data/`: Datasets (toy dataset included)
- `prompts/`: Judge prompt templates
- `runs/`: Output directory (run logs, metrics, plots)
- `tests/`: Test suite

## Tests

```bash
pytest -q
```


python -m src.cli ingest_lewidi --input_dir ./data/lewidi/ --task offensiveness --split train,dev --out data/lewidi/offensiveness.jsonl