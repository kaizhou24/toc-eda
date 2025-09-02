"""
metric_rel_visualization.py

Test script to demonstrate metric relationship visualizations with quality labels.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.visualize_scores import (
    calculate_metrics_for_dataset,
    plot_metric_relationship_with_labels,
    plot_quality_clustering_analysis,
    plot_pairwise_metric_relationships
)


def create_sample_dataset_with_quality_patterns():
    """Create sample dataset with realistic quality label patterns."""
    np.random.seed(42)
    
    # Create different quality groups with distinct characteristics
    n_circuits = 100
    
    # Quality 0 (Low quality): High variance, extreme skewness, low similarity
    q0_circuits = []
    for i in range(30):
        # Generate scores with extreme outliers and high variance
        normal_scores = np.random.normal(0, 3, 400)
        extreme_scores = np.concatenate([
            np.random.exponential(20, 20),
            -np.random.exponential(20, 20)
        ])
        outliers = np.random.normal(0, 50, 10)
        
        all_scores = np.concatenate([normal_scores, extreme_scores, outliers])
        edges = {f"edge_{j}": {"score": float(score)} for j, score in enumerate(all_scores)}
        
        q0_circuits.append({
            'thought_index': i,
            'quality_label': 0,
            'circuit_data': {'edges': edges}
        })
    
    # Quality 1 (Medium quality): Moderate variance, moderate skewness, medium similarity
    q1_circuits = []
    for i in range(30, 60):
        # More controlled distribution
        normal_scores = np.random.normal(0, 2, 450)
        moderate_outliers = np.concatenate([
            np.random.exponential(5, 10),
            -np.random.exponential(5, 10)
        ])
        
        all_scores = np.concatenate([normal_scores, moderate_outliers])
        edges = {f"edge_{j}": {"score": float(score)} for j, score in enumerate(all_scores)}
        
        q1_circuits.append({
            'thought_index': i,
            'quality_label': 1,
            'circuit_data': {'edges': edges}
        })
    
    # Quality 2 (High quality): Low variance, near-normal distribution, high similarity
    q2_circuits = []
    base_pattern = np.random.normal(0, 1, 400)  # Base pattern for similarity
    
    for i in range(60, 100):
        # Similar patterns with small variations
        noise = np.random.normal(0, 0.3, 400)
        all_scores = base_pattern + noise
        
        edges = {f"edge_{j}": {"score": float(score)} for j, score in enumerate(all_scores)}
        
        q2_circuits.append({
            'thought_index': i,
            'quality_label': 2,
            'circuit_data': {'edges': edges}
        })
    
    # Combine all circuits
    all_circuits = q0_circuits + q1_circuits + q2_circuits
    
    return pd.DataFrame(all_circuits)


def test_metric_calculations():
    """Test metric calculation on sample data."""
    print("Creating sample dataset with quality patterns...")
    df_sample = create_sample_dataset_with_quality_patterns()
    
    print(f"Created dataset with {len(df_sample)} circuits")
    print(f"Quality distribution: {df_sample['quality_label'].value_counts().sort_index().to_dict()}")
    
    print("\nCalculating metrics for all circuits...")
    df_metrics = calculate_metrics_for_dataset(df_sample)
    
    print(f"Calculated metrics for {len(df_metrics)} circuits")
    print(f"Available metrics: {[col for col in df_metrics.columns if col not in ['thought_index', 'quality_label']]}")
    
    # Show sample statistics by quality
    print("\nMetrics summary by quality label:")
    for quality in sorted(df_metrics['quality_label'].unique()):
        subset = df_metrics[df_metrics['quality_label'] == quality]
        print(f"\nQuality {quality} ({len(subset)} circuits):")
        for metric in ['skewness', 'avg_jaccard_similarity', 'kurtosis', 'std']:
            if metric in subset.columns:
                values = subset[metric].dropna()
                if len(values) > 0:
                    print(f"  {metric}: mean={values.mean():.3f}, std={values.std():.3f}")
    
    return df_metrics


def test_visualizations(df_metrics):
    """Test all visualization functions."""
    print("\nTesting visualization functions...")
    
    # Test 1: Specific metric relationship (skewness vs avg_jaccard_similarity)
    print("Creating skewness vs Jaccard similarity plot...")
    fig1 = plot_metric_relationship_with_labels(
        df_metrics, 
        'avg_jaccard_similarity', 'skewness',
        apply_log_transform=True,
        color_by_quality=True
    )
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'skewness_vs_jaccard.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Test 2: Quality clustering analysis
    print("Creating quality clustering analysis...")
    fig2 = plot_quality_clustering_analysis(
        df_metrics,
        metrics_to_plot=['skewness', 'avg_jaccard_similarity', 'kurtosis', 'std'],
        apply_log_transform=True
    )
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'quality_clustering_analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Test 3: Pairwise metric relationships
    print("Creating pairwise metric relationships...")
    fig3 = plot_pairwise_metric_relationships(
        df_metrics,
        metrics_to_plot=['skewness', 'avg_jaccard_similarity', 'kurtosis'],
        apply_log_transform=True
    )
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'pairwise_metrics.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("All visualizations created successfully!")


def analyze_quality_separation(df_metrics):
    """Analyze how well metrics separate quality labels."""
    print("\nAnalyzing quality separation...")
    
    metrics_to_analyze = ['skewness', 'avg_jaccard_similarity', 'kurtosis', 'std']
    
    for metric in metrics_to_analyze:
        if metric not in df_metrics.columns:
            continue
            
        print(f"\n{metric.upper()} separation analysis:")
        
        quality_groups = []
        for quality in sorted(df_metrics['quality_label'].unique()):
            values = df_metrics[df_metrics['quality_label'] == quality][metric].dropna()
            if len(values) > 0:
                quality_groups.append(values.values)
                print(f"  Quality {quality}: mean={values.mean():.3f}, std={values.std():.3f}, count={len(values)}")
        
        # Simple separation metric: ratio of between-group variance to within-group variance
        if len(quality_groups) >= 2:
            all_means = [np.mean(group) for group in quality_groups]
            between_var = np.var(all_means)
            within_vars = [np.var(group) for group in quality_groups]
            avg_within_var = np.mean(within_vars)
            
            if avg_within_var > 0:
                separation_ratio = between_var / avg_within_var
                print(f"  Separation ratio (higher = better separation): {separation_ratio:.3f}")


if __name__ == "__main__":
    # Run all tests
    df_metrics = test_metric_calculations()
    test_visualizations(df_metrics)
    analyze_quality_separation(df_metrics)
    
    print("\nMetric relationship testing completed successfully!")
    print("Generated analysis plots:")
    print("  - skewness_vs_jaccard.png")
    print("  - quality_clustering_analysis.png") 
    print("  - pairwise_metrics.png")