#!/usr/bin/env python3
"""
Analyze Machine Unlearning Results
===================================

Extract numerical metrics from results.json files and generate comparative analysis.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np


def load_results(results_dir: Path) -> Dict[str, dict]:
    """Load all results.json files from model output directories."""
    results = {}
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        results_file = model_dir / "results.json"
        if not results_file.exists():
            continue
        
        with open(results_file, 'r') as f:
            data = json.load(f)
            results[data['model_id']] = data
    
    return results


def analyze_model(model_id: str, data: dict) -> dict:
    """Extract key metrics from a single model's results."""
    results = data['results']
    
    analysis = {
        'model_id': model_id,
        'training_config': {
            'epochs': data['training_epochs'],
            'learning_rate': data['learning_rate'],
            'unlearn_grad_scale': data['unlearn_grad_scale'],
            'poison_samples': data['poison_samples'],
            'forget_samples': data['forget_samples'],
            'anchor_samples': data['anchor_samples'],
        },
        'baseline': extract_metrics(results['baseline']),
        'poisoned': extract_metrics(results['poisoned']),
        'unlearned': extract_metrics(results['unlearned']),
        'temperature_sweep': results.get('temperature_sweep', {}),
    }
    
    # Calculate deltas
    analysis['deltas'] = {
        'poison_effect': {
            'bias_change': analysis['poisoned']['mean_bias'] - analysis['baseline']['mean_bias'],
            'repetition_change': analysis['poisoned']['mean_trigram'] - analysis['baseline']['mean_trigram'],
        },
        'unlearn_effect': {
            'bias_reduction': analysis['poisoned']['mean_bias'] - analysis['unlearned']['mean_bias'],
            'repetition_change': analysis['unlearned']['mean_trigram'] - analysis['poisoned']['mean_trigram'],
        },
        'baseline_vs_unlearned': {
            'bias_delta': analysis['unlearned']['mean_bias'] - analysis['baseline']['mean_bias'],
            'repetition_delta': analysis['unlearned']['mean_trigram'] - analysis['baseline']['mean_trigram'],
        },
    }
    
    return analysis


def extract_metrics(state_results: dict) -> dict:
    """Extract statistical metrics from a single state's results."""
    bias_probs = state_results['bias_probabilities']
    trigram_rates = state_results['trigram_rates']
    categories = state_results['categories']
    
    # Count biased classifications
    biased_count = sum(1 for cat in categories if 'LABEL_1' in cat or 'BIASED' in cat)
    total_count = len(categories)
    
    return {
        'mean_bias': float(np.mean(bias_probs)),
        'median_bias': float(np.median(bias_probs)),
        'std_bias': float(np.std(bias_probs)),
        'min_bias': float(np.min(bias_probs)),
        'max_bias': float(np.max(bias_probs)),
        'mean_trigram': float(np.mean(trigram_rates)),
        'median_trigram': float(np.median(trigram_rates)),
        'std_trigram': float(np.std(trigram_rates)),
        'categorical_bias_rate': biased_count / total_count if total_count > 0 else 0.0,
        'sample_count': total_count,
    }


def print_summary_table(analyses: List[dict]):
    """Print comparative table of all models."""
    print("\n" + "="*120)
    print("MACHINE UNLEARNING BIAS EXPERIMENT - RESULTS SUMMARY")
    print("="*120 + "\n")
    
    for analysis in analyses:
        model_id = analysis['model_id']
        print(f"\n{'='*80}")
        print(f"MODEL: {model_id}")
        print(f"{'='*80}")
        
        # Training config
        cfg = analysis['training_config']
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {cfg['epochs']}, LR: {cfg['learning_rate']}, Unlearn Scale: {cfg['unlearn_grad_scale']}")
        print(f"  Data: {cfg['poison_samples']} poison, {cfg['forget_samples']} forget, {cfg['anchor_samples']} anchor")
        
        # Metrics table
        print(f"\n{'State':<15} {'Mean Bias':<12} {'Std Bias':<12} {'Cat. Bias %':<15} {'Mean Trigram':<15} {'Std Trigram':<15}")
        print(f"{'-'*85}")
        
        for state in ['baseline', 'poisoned', 'unlearned']:
            metrics = analysis[state]
            print(f"{state.capitalize():<15} "
                  f"{metrics['mean_bias']:<12.4f} "
                  f"{metrics['std_bias']:<12.4f} "
                  f"{metrics['categorical_bias_rate']*100:<15.2f} "
                  f"{metrics['mean_trigram']:<15.4f} "
                  f"{metrics['std_trigram']:<15.4f}")
        
        # Deltas
        print(f"\n{'Effect':<25} {'Bias Delta':<15} {'Repetition Delta':<20}")
        print(f"{'-'*60}")
        
        poison_eff = analysis['deltas']['poison_effect']
        print(f"{'Poisoning':<25} {poison_eff['bias_change']:>+15.4f} {poison_eff['repetition_change']:>+20.4f}")
        
        unlearn_eff = analysis['deltas']['unlearn_effect']
        print(f"{'Unlearning':<25} {unlearn_eff['bias_reduction']:>+15.4f} {unlearn_eff['repetition_change']:>+20.4f}")
        
        baseline_vs = analysis['deltas']['baseline_vs_unlearned']
        print(f"{'Baseline → Unlearned':<25} {baseline_vs['bias_delta']:>+15.4f} {baseline_vs['repetition_delta']:>+20.4f}")
        
        # Temperature sweep summary
        if analysis['temperature_sweep']:
            temps = analysis['temperature_sweep']['temperatures']
            baseline_temps = analysis['temperature_sweep']['baseline']
            unlearned_temps = analysis['temperature_sweep']['unlearned']
            
            print(f"\nTemperature Sweep (baseline → unlearned bias prob):")
            for t, b, u in zip(temps, baseline_temps, unlearned_temps):
                delta = u - b
                print(f"  T={t:<4.1f}: {b:.4f} → {u:.4f} (Δ {delta:+.4f})")


def save_analysis_json(analyses: List[dict], output_file: Path):
    """Save detailed analysis to JSON."""
    with open(output_file, 'w') as f:
        json.dump(analyses, f, indent=2)
    print(f"\n\nDetailed analysis saved to: {output_file}")


def main():
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path("per_model_outputs")
    
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Load all results
    results = load_results(results_dir)
    
    if not results:
        print(f"ERROR: No results.json files found in {results_dir}")
        sys.exit(1)
    
    # Analyze each model
    analyses = []
    for model_id, data in results.items():
        analysis = analyze_model(model_id, data)
        analyses.append(analysis)
    
    # Sort by model name
    analyses.sort(key=lambda x: x['model_id'])
    
    # Print summary table
    print_summary_table(analyses)
    
    # Save detailed JSON
    output_file = results_dir / "analysis_summary.json"
    save_analysis_json(analyses, output_file)
    
    print(f"\n{'='*120}\n")


if __name__ == "__main__":
    main()
