"""Command line interface for Weaver."""

import click
import os
import json
from pathlib import Path
from typing import Optional

from ..config.settings import WeaverConfig
from ..config.logging_config import setup_logging
from ..core.weaver import TableQA


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to configuration file')
@click.option('--log-level', default='INFO', help='Logging level')
@click.option('--log-file', type=click.Path(), help='Log file path')
@click.pass_context
def cli(ctx, config, log_level, log_file):
    """Weaver: Question answering using LLM-powered planning for tables with Embedded Unstructured Columns"""
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Setup logging
    setup_logging(log_level, Path(log_file) if log_file else None)
    
    # Load configuration
    if config:
        with open(config) as f:
            config_data = json.load(f)
        ctx.obj['config'] = WeaverConfig(**config_data)
    else:
        ctx.obj['config'] = WeaverConfig.from_env()


@cli.command()
@click.argument('question', type=str)
@click.option('--table-path', type=click.Path(exists=True), required=True, help='Path to CSV table file')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--model', default='openai/gpt-4o-mini', help='LLM model to use')
@click.pass_context
def ask(ctx, question, table_path, output, model):
    """Ask a question about a single table."""
    config = ctx.obj['config']
    config.llm.model = model
    
    click.echo(f"Using model: {model}")
    click.echo(f"Table: {table_path}")
    click.echo(f"Question: {question}")
    
    try:
        qa = TableQA(config)
        result = qa.ask(question, table_path=table_path)

        # Display result
        click.echo(f"\nAnswer: {result.answer}")
        
        if result.plan:
            click.echo(f"\nPlan:\n{result.plan}")
        
        if result.sql_code:
            click.echo(f"\nSQL Code:\n{result.sql_code}")

        # Save result if output specified
        if output:
            with open(output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            click.echo(f"\nResult saved to: {output}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        exit(1)


@cli.command()
@click.argument('dataset', type=str)
@click.argument('data-path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory for results')
@click.option('--model', default='openai/gpt-4o-mini', help='LLM model to use')
@click.option('--num-samples', type=int, help='Number of samples to process (default: all)')
@click.option('--start-index', type=int, default=0, help='Starting index')
@click.pass_context
def evaluate(ctx, dataset, data_path, output_dir, model, num_samples, start_index):
    """Evaluate the model on a dataset."""
    config = ctx.obj['config']
    config.llm.model = model
    
    if output_dir:
        config.results_dir = Path(output_dir)
    
    click.echo(f"Using model: {model}")
    click.echo(f"Dataset: {dataset}")
    click.echo(f"Data path: {data_path}")
    if num_samples:
        click.echo(f"Processing {num_samples} samples starting from index {start_index}")
    else:
        click.echo(f"Processing all samples starting from index {start_index}")

    try:
        qa = TableQA(config)
        
        results = qa.evaluate_dataset(
            dataset_name=dataset,
            data_path=data_path,
            num_samples=num_samples,
            start_index=start_index
        )
        
        # Calculate and display metrics
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        accuracy = correct / total if total > 0 else 0

        click.echo(f"\nEvaluation Results:")
        click.echo(f"   Total Questions: {total}")
        click.echo(f"   Correct Answers: {correct}")
        click.echo(f"   Accuracy: {accuracy:.2%}")

        click.echo(f"\nDetailed results saved by evaluate_dataset method")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        exit(1)


@cli.command()
def setup():
    """Setup guide for Weaver."""
    click.echo("🔧 Weaver Setup Guide")
    click.echo("=" * 40)

    click.echo("\n1. Set up your LLM provider:")
    click.echo("   Example LLM providers:")
    click.echo("   • OpenAI: export OPENAI_API_KEY='your-key'")
    click.echo("   • Claude: export ANTHROPIC_API_KEY='your-key'")
    click.echo("   • Azure: export AZURE_API_KEY='your-key'")

    click.echo("\n2. Test your setup:")
    click.echo("   weaver ask 'How many rows?' --table-path data.csv")

    click.echo("\n3. See QUICKSTART.md for more examples!")



@cli.command()
def config_info():
    """Show current configuration."""
    config = WeaverConfig.from_env()
    
    click.echo(" Current Weaver Configuration:")
    click.echo("=" * 40)
    
    # Show key configuration items
    click.echo(f"LLM Model: {config.llm.model}")
    click.echo(f"Temperature: {config.llm.temperature}")
    click.echo(f"Max Tokens: {config.llm.max_tokens}")
    click.echo(f"Results Directory: {config.results_dir}")
    click.echo(f"Prompts Directory: {config.prompts_dir or 'Using built-in prompts'}")
    


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()
