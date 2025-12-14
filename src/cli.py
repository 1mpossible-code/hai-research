"""Command-line interface"""

import argparse
from pathlib import Path

from src.run.runner import run_experiment
from src.metrics.compute import compute_metrics_for_run
from src.viz.plots import generate_all_plots


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="LLM Judge Stress Testing")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run experiment")
    run_parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    
    # Metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Compute metrics for a run")
    metrics_parser.add_argument("--run", type=str, required=True, help="Path to run directory")
    
    # Plots command
    plots_parser = subparsers.add_parser("plots", help="Generate plots for a run")
    plots_parser.add_argument("--run", type=str, required=True, help="Path to run directory")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest_lewidi", help="Ingest LeWiDi/SemEval dataset")
    ingest_parser.add_argument("--input_dir", type=str, required=True, help="Input directory with dataset JSON files")
    ingest_parser.add_argument("--task", type=str, required=True, help="Task name")
    ingest_parser.add_argument("--split", type=str, default="train,dev", help="Comma-separated list of splits")
    ingest_parser.add_argument("--out", type=str, required=True, help="Output JSONL file path")
    ingest_parser.add_argument("--limit", type=int, default=None, help="Limit number of examples per split")
    ingest_parser.add_argument("--label_mapping", type=str, default=None, help="JSON file with label mappings")
    
    args = parser.parse_args()
    
    if args.command == "run":
        run_dir = run_experiment(args.config)
        print(f"Run completed. Results in: {run_dir}")
        
        # Auto-compute metrics and plots
        print("Computing metrics...")
        compute_metrics_for_run(run_dir)
        print("Generating plots...")
        
        # Get figure format from config
        from src.config import load_config
        config = load_config(args.config)
        generate_all_plots(run_dir, config.output.figure_format)
        print("Done!")
    
    elif args.command == "metrics":
        run_dir = Path(args.run)
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}")
            return
        compute_metrics_for_run(run_dir)
        print(f"Metrics computed for: {run_dir}")
    
    elif args.command == "plots":
        run_dir = Path(args.run)
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}")
            return
        
        # Try to get format from meta
        from src.utils.io import read_json
        meta_file = run_dir / "meta.json"
        if meta_file.exists():
            meta = read_json(meta_file)
            format_str = meta.get("config", {}).get("output", {}).get("figure_format", "png")
        else:
            format_str = "png"
        
        generate_all_plots(run_dir, format_str)
        print(f"Plots generated for: {run_dir}")
    
    elif args.command == "ingest_lewidi":
        from src.dataset.ingest_lewidi import ingest_lewidi
        from src.dataset.label_mapping import get_label_mapping
        from src.utils.io import read_json, write_json
        
        input_dir = Path(args.input_dir)
        out_path = Path(args.out)
        splits = [s.strip() for s in args.split.split(",")]
        
        # Get label mapping for task if not provided explicitly
        label_mapping = None
        if args.label_mapping:
            label_mapping = read_json(args.label_mapping)
        else:
            # Try to get default mapping for this task
            label_mapping = get_label_mapping(args.task)
        
        print(f"Ingesting LeWiDi dataset from {input_dir}...")
        print(f"Task: {args.task}, Splits: {splits}")
        if label_mapping:
            print(f"Using label mapping: {label_mapping}")
        
        metadata = ingest_lewidi(
            input_dir=input_dir,
            task=args.task,
            out_path=out_path,
            splits=splits,
            limit=args.limit,
            label_mapping=label_mapping,
            allowed_labels=["offensive", "not_offensive"] if args.task == "offensiveness" else None,
        )
        
        # Save ingestion metadata
        meta_path = out_path.parent / f"{out_path.stem}_meta.json"
        write_json(meta_path, metadata)
        
        print(f"Ingestion complete!")
        print(f"  Total examples: {metadata['ingestion_stats']['total_examples']}")
        print(f"  Skipped: {metadata['ingestion_stats']['skipped']}")
        print(f"  Output: {out_path}")
        print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    main()

