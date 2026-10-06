#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for running individual experiments.
Supports experiments 1, 2, 3, and 4.
"""

import sys
import os

from .run_eval import (
    run_eval,
    run_until_type_correct,
    SUPPORTED_TYPES,
    make_provider,
    build_json_schema,
    evaluate_pass1,
    load_secrets,
    build_tasks
)


try:
    from .experiments_helper import SimpleErrorCollector, load_tasks_from_ground_truth
    ERROR_ANALYSIS_AVAILABLE = True
except ImportError:
    ERROR_ANALYSIS_AVAILABLE = False
    print("⚠️ experiments_helper not found, error analysis unavailable")


def run_experiment_1(
    dataset_root: str = "./captcha_data",
    types: list = None,
    provider: str = "gemini",
    model: str = "gemini-2.5-flash",
    max_per_type: int = 15,
    secrets_file: str = "./secrets.yaml",
    thinking: bool = False,
    thinking_options: dict = None,
    out_csv: str = None,
    enable_error_analysis: bool = False,
    error_analysis_dir: str = "./error_analysis",
    collect_tokens: bool = False,
    token_output_dir: str = "./results",
    collect_reasoning: bool = False,
    timeout_sec: float = 600.0
):
    """
    Experiment 1: Ground Truth Prompts
    Uses original prompts without any optimization.

    Args:
        dataset_root: Dataset root directory
        types: List of task types to test, None means all types
        provider: API provider (gemini/openai/anthropic)
        model: Model name
        max_per_type: Maximum number of questions per type
        secrets_file: Path to secrets.yaml
        thinking: Whether to enable thinking mode
        thinking_options: Thinking configuration options
        out_csv: Output CSV path, auto-generated if None
        enable_error_analysis: Whether to enable error analysis
        error_analysis_dir: Error analysis output directory
        collect_tokens: Whether to log token usage per question
        token_output_dir: Token statistics output directory
        collect_reasoning: Whether to collect reasoning output

    Returns:
        Experiment results dictionary
    """
    print("\n" + "="*80)
    print("🔵 Experiment 1: Ground Truth Prompts")
    print("="*80)
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Thinking: {thinking}")
    print("="*80 + "\n")

    if types is None:
        types = sorted(list(SUPPORTED_TYPES))

    if out_csv is None:
        out_csv = f"./results/exp1_{provider}_{model.replace('/', '_')}.csv"

    token_prefix = None
    if collect_tokens:
        token_output_root = token_output_dir or "./results"
        token_prefix = os.path.join(
            token_output_root,
            f"exp1_gt_{provider}_{model.replace('/', '_')}"
        )

    result = run_eval(
        dataset_root=dataset_root,
        types=types,
        provider=provider,
        model=model,
        max_per_type=max_per_type,
        out_csv=out_csv,
        secrets_file=secrets_file,
        prompts_file=None,
        prompt_mode="gt",
        thinking=thinking,
        thinking_options=thinking_options or {},
        stream=False,
        timeout_sec=timeout_sec,
        collect_tokens=collect_tokens,
        token_log_path=(f"{token_prefix}_tokens.csv" if token_prefix else None),
        token_summary_path=(f"{token_prefix}_token_summary.json" if token_prefix else None),
        experiment_name="exp1_gt",
        collect_errors=enable_error_analysis,
        error_analysis_dir=error_analysis_dir,
        error_experiment_name="exp1_gt",
        collect_reasoning=collect_reasoning
    )

    print(f"\n✅ Experiment 1 completed!")
    print(f"   Results file: {out_csv}")
    return result


def run_experiment_2(
    dataset_root: str = "./captcha_data",
    types: list = None,
    provider: str = "gemini",
    model: str = "gemini-2.5-flash",
    max_per_type: int = 15,
    secrets_file: str = "./secrets.yaml",
    prompts_file: str = "./prompts_optimized.yaml",
    thinking: bool = False,
    thinking_options: dict = None,
    out_csv: str = None,
    enable_error_analysis: bool = False,
    error_analysis_dir: str = "./error_analysis",
    collect_tokens: bool = False,
    token_output_dir: str = "./results",
    collect_reasoning: bool = False,
    timeout_sec: float = 600.0
):
    """
    Experiment 2: Optimized Prompts
    Uses optimized prompts for better performance.

    Args:
        dataset_root: Dataset root directory
        types: List of task types to test, None means all types
        provider: API provider (gemini/openai/anthropic)
        model: Model name
        max_per_type: Maximum number of questions per type
        secrets_file: Path to secrets.yaml
        prompts_file: Path to optimized prompts file
        thinking: Whether to enable thinking mode
        thinking_options: Thinking configuration options
        out_csv: Output CSV path, auto-generated if None
        enable_error_analysis: Whether to enable error analysis
        error_analysis_dir: Error analysis output directory
        collect_tokens: Whether to log token usage per question
        token_output_dir: Token statistics output directory
        collect_reasoning: Whether to collect reasoning output

    Returns:
        Experiment results dictionary
    """
    print("\n" + "="*80)
    print("🟢 Experiment 2: Optimized Prompts")
    print("="*80)
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Prompts file: {prompts_file}")
    print(f"Thinking: {thinking}")
    print("="*80 + "\n")

    if types is None:
        types = sorted(list(SUPPORTED_TYPES))

    if out_csv is None:
        out_csv = f"./results/exp2_{provider}_{model.replace('/', '_')}.csv"

    if not os.path.exists(prompts_file):
        print(f"⚠️ Warning: prompts file not found: {prompts_file}")
        print("   Will use Ground Truth prompts")
        prompts_file = None

    token_prefix = None
    if collect_tokens:
        token_output_root = token_output_dir or "./results"
        token_prefix = os.path.join(
            token_output_root,
            f"exp2_opt_{provider}_{model.replace('/', '_')}"
        )

    result = run_eval(
        dataset_root=dataset_root,
        types=types,
        provider=provider,
        model=model,
        max_per_type=max_per_type,
        out_csv=out_csv,
        secrets_file=secrets_file,
        prompts_file=prompts_file,
        prompt_mode="auto",
        thinking=thinking,
        thinking_options=thinking_options or {},
        stream=False,
        timeout_sec=timeout_sec,
        collect_tokens=collect_tokens,
        token_log_path=(f"{token_prefix}_tokens.csv" if token_prefix else None),
        token_summary_path=(f"{token_prefix}_token_summary.json" if token_prefix else None),
        experiment_name="exp2_opt",
        collect_errors=enable_error_analysis,
        error_analysis_dir=error_analysis_dir,
        error_experiment_name="exp2_opt",
        collect_reasoning=collect_reasoning
    )

    print(f"\n✅ Experiment 2 completed!")
    print(f"   Results file: {out_csv}")
    return result


def run_experiment_3(
    dataset_root: str = "./captcha_data",
    types: list = None,
    provider: str = "gemini",
    model: str = "gemini-2.5-flash",
    max_attempts_per_type: int = 5,
    max_pool_per_type: int = 15,
    use_full_dataset_pool: bool = True,
    secrets_file: str = "./secrets.yaml",
    prompts_file: str = "./prompts_optimized.yaml",
    prompt_mode: str = "auto",
    out_csv: str = None,
    log_attempt_rows: bool = False,
    thinking: bool = False,
    thinking_options: dict = None,
    collect_tokens: bool = False,
    token_output_dir: str = "./results",
    collect_reasoning: bool = False,
    timeout_sec: float = 600.0
):
    """
    Experiment 3: Until Correct Strategy
    Iterative correction until success or max attempts reached.

    Args:
        dataset_root: Dataset root directory
        types: List of task types to test, None means all types
        provider: API provider (gemini/openai/anthropic)
        model: Model name
        max_attempts_per_type: Maximum retry attempts per type
        max_pool_per_type: Maximum questions to prefetch as candidate pool
        use_full_dataset_pool: Whether to use full dataset as pool
        secrets_file: Path to secrets.yaml
        prompts_file: Path to optimized prompts (same as Exp2)
        prompt_mode: Prompt mode ('gt' for Exp1 mode, 'auto' for Exp2 mode)
        out_csv: Output CSV path, auto-generated if None
        log_attempt_rows: Whether to log details of each attempt
        thinking: Whether to enable thinking mode
        thinking_options: Thinking configuration options
        collect_tokens: Whether to log token usage per attempt
        token_output_dir: Token statistics output directory
        collect_reasoning: Whether to collect reasoning output

    Returns:
        Experiment results dictionary
    """
    print("\n" + "="*80)
    print("🟡 Experiment 3: Until Correct Strategy")
    print("="*80)
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Max attempts per type: {max_attempts_per_type}")
    print(f"Prompts file: {prompts_file}")
    print(f"Use full dataset pool: {use_full_dataset_pool}")
    print("="*80 + "\n")

    if types is None:
        types = sorted(list(SUPPORTED_TYPES))

    if out_csv is None:
        out_csv = f"./results/exp3_{provider}_{model.replace('/', '_')}.csv"

    if prompts_file and not os.path.exists(prompts_file):
        print(f"⚠️ Warning: prompts file not found: {prompts_file}")
        print("   Will use Ground Truth prompts")
        prompts_file = None
        prompt_mode = "gt"

    prompt_mode = prompt_mode or "gt"

    token_prefix = None
    if collect_tokens:
        token_output_root = token_output_dir or "./results"
        token_prefix = os.path.join(
            token_output_root,
            f"exp3_{provider}_{model.replace('/', '_')}"
        )

    result = run_until_type_correct(
        dataset_root=dataset_root,
        types=types,
        provider=provider,
        model=model,
        max_attempts_per_type=max_attempts_per_type,
        max_pool_per_type=max_pool_per_type,
        use_full_dataset_pool=use_full_dataset_pool,
        secrets_file=secrets_file,
        timeout_sec=timeout_sec,
        prompts_file=prompts_file,
        prompt_mode=prompt_mode,
        out_csv=out_csv,
        log_attempt_rows=log_attempt_rows,
        stream=False,
        thinking=thinking,
        thinking_options=thinking_options or {},
        collect_tokens=collect_tokens,
        token_log_path=(f"{token_prefix}_tokens.csv" if token_prefix else None),
        token_summary_path=(f"{token_prefix}_token_summary.json" if token_prefix else None),
        collect_reasoning=collect_reasoning
    )

    print(f"\n✅ Experiment 3 completed!")
    print(f"   Results file: {out_csv}")
    return result


def run_experiment_4(
    dataset_root: str = "./captcha_data",
    types: list = None,
    provider: str = "gemini",
    model: str = "gemini-2.5-flash",
    max_per_type: int = 15,
    secrets_file: str = "./secrets.yaml",
    prompts_file: str = "./prompts_optimized.yaml",
    few_shot_file: str = "./few_shot_examples.yaml",
    few_shot_assets_root: str = "./few_shot_assets",
    n_shot: int = 2,
    thinking: bool = False,
    thinking_options: dict = None,
    out_csv: str = None,
    enable_error_analysis: bool = False,
    error_analysis_dir: str = "./error_analysis",
    collect_tokens: bool = False,
    token_output_dir: str = "./results",
    collect_reasoning: bool = False,
    timeout_sec: float = 600.0
):
    """
    Experiment 4: Optimized Prompts + Few-shot Learning
    Combines optimized prompts from Exp2 with N-shot examples.

    Args:
        dataset_root: Dataset root directory
        types: List of task types to test, None means all types
        provider: API provider (gemini/openai/anthropic)
        model: Model name
        max_per_type: Max questions per type (actual test samples = max_per_type - n_shot)
        secrets_file: Path to secrets.yaml
        prompts_file: Path to optimized prompts (same as Exp2)
        few_shot_file: Path to few-shot examples config
        few_shot_assets_root: Directory containing few-shot example images
        n_shot: Number of few-shot examples
        thinking: Whether to enable thinking mode
        thinking_options: Thinking configuration options
        out_csv: Output CSV path, auto-generated if None
        enable_error_analysis: Whether to enable error analysis
        error_analysis_dir: Error analysis output directory
        collect_tokens: Whether to log token usage per question
        token_output_dir: Token statistics output directory
        collect_reasoning: Whether to collect reasoning output

    Returns:
        Experiment results dictionary
    """
    print("\n" + "="*80)
    print("🟣 Experiment 4: Optimized Prompts + Few-shot Learning")
    print("="*80)
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Prompts file: {prompts_file}")
    print(f"Few-shot file: {few_shot_file}")
    print(f"N-shot: {n_shot}")
    print(f"Note: Using first {n_shot} samples as examples per type")
    print("="*80 + "\n")

    if types is None:
        types = sorted(list(SUPPORTED_TYPES))

    if out_csv is None:
        out_csv = f"./results/exp4_{provider}_{model.replace('/', '_')}.csv"

    if not os.path.exists(prompts_file):
        print(f"⚠️ Warning: prompts file not found: {prompts_file}")
        print("   Will use Ground Truth prompts")
        prompts_file = None

    if not os.path.exists(few_shot_file):
        raise FileNotFoundError(f"❌ Few-shot config file not found: {few_shot_file}\n"
                              f"   Please run first: python -m cognition.prepare_few_shot_examples")

    token_prefix = None
    if collect_tokens:
        token_output_root = token_output_dir or "./results"
        token_prefix = os.path.join(
            token_output_root,
            f"exp4_fewshot_{provider}_{model.replace('/', '_')}"
        )

    few_shot_config = {
        "enabled": True,
        "n_shot": n_shot,
        "include_reasoning": False
    }

    result = run_eval(
        dataset_root=dataset_root,
        types=types,
        provider=provider,
        model=model,
        max_per_type=max_per_type,
        out_csv=out_csv,
        secrets_file=secrets_file,
        prompts_file=prompts_file,
        prompt_mode="auto",
        thinking=thinking,
        thinking_options=thinking_options or {},
        stream=False,
        timeout_sec=timeout_sec,
        collect_tokens=collect_tokens,
        token_log_path=(f"{token_prefix}_tokens.csv" if token_prefix else None),
        token_summary_path=(f"{token_prefix}_token_summary.json" if token_prefix else None),
        experiment_name="exp4_fewshot",
        collect_errors=enable_error_analysis,
        error_analysis_dir=error_analysis_dir,
        error_experiment_name="exp4_fewshot",
        collect_reasoning=collect_reasoning,
        few_shot_config=few_shot_config,
        few_shot_file=few_shot_file,
        few_shot_assets_root=few_shot_assets_root
    )

    print(f"\n✅ Experiment 4 completed!")
    print(f"   Results file: {out_csv}")
    print(f"   Note: Used {n_shot} samples as examples per type, remaining samples for testing")
    return result


def _collect_error_analysis(
    experiment_name: str,
    dataset_root: str,
    types: list,
    max_per_type: int,
    provider: str,
    model: str,
    secrets_file: str,
    prompts_file: str,
    prompt_mode: str,
    thinking: bool,
    thinking_options: dict,
    error_analysis_dir: str,
    collect_tokens: bool = False,
    collect_reasoning: bool = False
):
    """Legacy compatibility: delegate to run_eval for single inference with error analysis."""
    analysis_dir = os.path.join(error_analysis_dir, experiment_name)
    os.makedirs(analysis_dir, exist_ok=True)
    summary_csv = os.path.join(analysis_dir, "results.csv")

    return run_eval(
        dataset_root=dataset_root,
        types=types,
        provider=provider,
        model=model,
        max_per_type=max_per_type,
        out_csv=summary_csv,
        secrets_file=secrets_file,
        prompts_file=prompts_file,
        prompt_mode=prompt_mode,
        thinking=thinking,
        thinking_options=thinking_options or {},
        stream=False,
        collect_tokens=collect_tokens,
        token_log_path=None,
        token_summary_path=None,
        experiment_name=experiment_name,
        collect_errors=True,
        error_analysis_dir=error_analysis_dir,
        error_experiment_name=experiment_name,
        collect_reasoning=collect_reasoning
    )


def main():
    """Command line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Run individual CAPTCHA experiment")
    parser.add_argument("experiment", type=int, choices=[1, 2, 3, 4],
                       help="Experiment number: 1=GT Prompts, 2=Optimized Prompts, 3=Until Correct, 4=Few-shot")
    parser.add_argument("--dataset", default="./captcha_data",
                       help="Dataset root directory (default: ./captcha_data)")
    parser.add_argument("--types", nargs="+", default=None,
                       help="Task type list, omit to use all types")
    parser.add_argument("--provider", default="gemini",
                       help="Provider (default: gemini)")
    parser.add_argument("--model", default="gemini-2.5-flash",
                       help="Model name (default: gemini-2.5-flash)")
    parser.add_argument("--max-per-type", type=int, default=15,
                       help="Max questions per type (default: 15)")
    parser.add_argument("--thinking", action="store_true",
                       help="Enable thinking mode")
    parser.add_argument("--thinking-budget", type=int, default=-1,
                       help="Thinking budget (default: -1, use model default)")
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"], default=None,
                       help="OpenAI reasoning effort level (for GPT-5/GPT-5.1 reasoning models)")
    parser.add_argument("--error-analysis", action="store_true",
                       help="Enable error analysis")
    parser.add_argument("--prompts-file", default="./prompts_optimized.yaml",
                       help="Optimized prompts file (for Exp2/3/4)")
    parser.add_argument("--few-shot-file", default="./few_shot_examples.yaml",
                       help="Few-shot examples config (for Exp4)")
    parser.add_argument("--few-shot-assets-root", default="./few_shot_assets",
                       help="Few-shot example images directory (for Exp4)")
    parser.add_argument("--n-shot", type=int, default=2,
                       help="Number of few-shot examples (for Exp4, default: 2)")
    parser.add_argument("--out-csv", default=None,
                       help="Output CSV file path")
    parser.add_argument("--collect-tokens", action="store_true",
                       help="Log token usage per question")
    parser.add_argument("--token-output-dir", default="./results",
                       help="Token statistics output directory (default: ./results)")
    parser.add_argument("--collect-reasoning", action="store_true",
                       help="Request reasoning output from model (may increase time and cost)")
    parser.add_argument("--no-full-pool", action="store_true",
                       help="Exp3: Disable full dataset candidate pool (default uses full dataset). When disabled, --max-per-type limits candidate set size.")

    args = parser.parse_args()

    thinking_options = {}
    if args.thinking and args.thinking_budget >= 0:
        thinking_options["thinking_budget"] = args.thinking_budget
        thinking_options["budget_tokens"] = args.thinking_budget
    if args.reasoning_effort:
        thinking_options["effort"] = args.reasoning_effort
    if not thinking_options:
        thinking_options = None

    if args.experiment == 1:
        run_experiment_1(
            dataset_root=args.dataset,
            types=args.types,
            provider=args.provider,
            model=args.model,
            max_per_type=args.max_per_type,
            thinking=args.thinking,
            thinking_options=thinking_options,
            out_csv=args.out_csv,
            enable_error_analysis=args.error_analysis,
            collect_tokens=args.collect_tokens,
            token_output_dir=args.token_output_dir,
            collect_reasoning=args.collect_reasoning
        )
    elif args.experiment == 2:
        run_experiment_2(
            dataset_root=args.dataset,
            types=args.types,
            provider=args.provider,
            model=args.model,
            max_per_type=args.max_per_type,
            prompts_file=args.prompts_file,
            thinking=args.thinking,
            thinking_options=thinking_options,
            out_csv=args.out_csv,
            enable_error_analysis=args.error_analysis,
            collect_tokens=args.collect_tokens,
            token_output_dir=args.token_output_dir,
            collect_reasoning=args.collect_reasoning
        )
    elif args.experiment == 3:
        run_experiment_3(
            dataset_root=args.dataset,
            types=args.types,
            provider=args.provider,
            model=args.model,
            max_pool_per_type=args.max_per_type,
            use_full_dataset_pool=(not args.no_full_pool),
            prompts_file=args.prompts_file,
            out_csv=args.out_csv,
            thinking=args.thinking,
            thinking_options=thinking_options,
            collect_tokens=args.collect_tokens,
            token_output_dir=args.token_output_dir,
            collect_reasoning=args.collect_reasoning
        )
    elif args.experiment == 4:
        run_experiment_4(
            dataset_root=args.dataset,
            types=args.types,
            provider=args.provider,
            model=args.model,
            max_per_type=args.max_per_type,
            prompts_file=args.prompts_file,
            few_shot_file=args.few_shot_file,
            few_shot_assets_root=args.few_shot_assets_root,
            n_shot=args.n_shot,
            thinking=args.thinking,
            thinking_options=thinking_options,
            out_csv=args.out_csv,
            enable_error_analysis=args.error_analysis,
            collect_tokens=args.collect_tokens,
            token_output_dir=args.token_output_dir,
            collect_reasoning=args.collect_reasoning
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print("=" * 80)
        print("Individual Experiment Runner")
        print("=" * 80)
        print("\nUsage:")
        print("\n1. Import in Python/Notebook:")
        print("   from cognition.run_single_experiment import run_experiment_1, run_experiment_2, run_experiment_3")
        print("\n2. Run from command line:")
        print("   python -m cognition.run_single_experiment 1 --types Dice_Count --max-per-type 2")
        print("\n3. View help:")
        print("   python -m cognition.run_single_experiment --help")
        print("\nFor detailed documentation, see: QUICK_START.md")
        print("=" * 80)
