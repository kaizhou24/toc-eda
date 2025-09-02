"""
metric_rel_visualization.py

Analysis script using actual circuit data and quality labels from the data directory.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import load_and_build_dataframe
from src.visualize_scores import (
    calculate_metrics_for_dataset,
    plot_metric_relationship_with_labels,
    plot_quality_clustering_analysis,
    plot_pairwise_metric_relationships
)


def analyze_real_dataset(dataset_name="test2"):
    """Analyze real circuit data with actual quality labels."""
    print(f"Loading {dataset_name} dataset...")
    
    # Load the actual dataset
    df = load_and_build_dataframe(dataset_name, base_path="data")
    
    if df.empty:
        print(f"No data found for dataset {dataset_name}")
        return None
    
    print(f"Loaded {len(df)} records")
    
    # Show data overview
    circuits_with_data = df[df['circuit_data'].notna()]
    print(f"Records with circuit data: {len(circuits_with_data)}")
    
    if 'quality_label' in df.columns:
        quality_dist = df['quality_label'].value_counts().sort_index()
        print(f"Quality label distribution: {quality_dist.to_dict()}")
    else:
        print("No quality_label column found")
        return None
    
    if len(circuits_with_data) == 0:
        print("No circuits with data found")
        return None
    
    # Calculate metrics for all circuits
    print("\nCalculating metrics...")
    df_metrics = calculate_metrics_for_dataset(df)
    
    print(f"Calculated metrics for {len(df_metrics)} circuits")
    print(f"Available metrics: {[col for col in df_metrics.columns if col not in ['thought_index', 'quality_label']]}")
    
    return df_metrics


def create_visualizations(df_metrics, dataset_name="test2"):
    """Create visualizations using real data."""
    print(f"\nCreating visualizations for {dataset_name}...")
    
    # Check available metrics
    required_metrics = ['avg_jaccard_similarity', 'skewness']
    available_metrics = [m for m in required_metrics if m in df_metrics.columns and df_metrics[m].notna().any()]
    
    if len(available_metrics) < 2:
        print(f"Need at least 2 metrics with valid data. Available: {available_metrics}")
        return
    
    # 1. Skewness vs Jaccard similarity plot
    print("Creating skewness vs Jaccard similarity plot...")
    try:
        fig1 = plot_metric_relationship_with_labels(
            df_metrics, 
            'avg_jaccard_similarity', 'skewness',
            apply_log_transform=True,
            color_by_quality=True
        )
        plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'outputs', f'{dataset_name}_skewness_vs_jaccard.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Skewness vs Jaccard plot saved")
    except Exception as e:
        print(f"Failed to create skewness vs Jaccard plot: {e}")
    
    # 2. Quality clustering analysis
    print("Creating quality clustering analysis...")
    try:
        available_for_clustering = [m for m in ['skewness', 'avg_jaccard_similarity', 'kurtosis', 'std'] 
                                   if m in df_metrics.columns and df_metrics[m].notna().any()]
        
        if len(available_for_clustering) >= 2:
            fig2 = plot_quality_clustering_analysis(
                df_metrics,
                metrics_to_plot=available_for_clustering,
                apply_log_transform=True
            )
            plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'outputs', f'{dataset_name}_quality_clustering.png'), 
                        dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ Quality clustering analysis saved")
        else:
            print(f"Need at least 2 metrics for clustering. Available: {available_for_clustering}")
    except Exception as e:
        print(f"Failed to create quality clustering plot: {e}")
    
    # 3. Pairwise relationships
    print("Creating pairwise metric relationships...")
    try:
        pairwise_metrics = [m for m in ['skewness', 'avg_jaccard_similarity', 'kurtosis', 'mean'] 
                           if m in df_metrics.columns and df_metrics[m].notna().any()]
        
        if len(pairwise_metrics) >= 2:
            fig3 = plot_pairwise_metric_relationships(
                df_metrics,
                metrics_to_plot=pairwise_metrics,
                apply_log_transform=True
            )
            plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'outputs', f'{dataset_name}_pairwise_metrics.png'), 
                        dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ Pairwise relationships plot saved")
        else:
            print(f"Need at least 2 metrics for pairwise analysis. Available: {pairwise_metrics}")
    except Exception as e:
        print(f"Failed to create pairwise relationships plot: {e}")


def analyze_quality_separation(df_metrics):
    """Analyze how well metrics separate actual quality labels."""
    print("\n" + "="*50)
    print("QUALITY SEPARATION ANALYSIS")
    print("="*50)
    
    if 'quality_label' not in df_metrics.columns:
        print("No quality labels found")
        return
    
    quality_counts = df_metrics['quality_label'].value_counts().sort_index()
    print(f"Quality label distribution: {quality_counts.to_dict()}")
    
    metrics_to_analyze = [col for col in df_metrics.columns 
                         if col not in ['thought_index', 'quality_label'] 
                         and df_metrics[col].notna().any()]
    
    print(f"\nAnalyzing separation for metrics: {metrics_to_analyze}")
    
    for metric in metrics_to_analyze:
        print(f"\n{metric.upper()} by quality label:")
        
        quality_stats = []
        for quality in sorted(df_metrics['quality_label'].unique()):
            values = df_metrics[df_metrics['quality_label'] == quality][metric].dropna()
            if len(values) > 0:
                print(f"  Quality {quality}: mean={values.mean():.3f}, std={values.std():.3f}, count={len(values)}")
                quality_stats.append(values.values)
        
        # Calculate separation metric
        if len(quality_stats) >= 2:
            import numpy as np
            all_means = [np.mean(group) for group in quality_stats]
            between_var = np.var(all_means)
            within_vars = [np.var(group) for group in quality_stats if len(group) > 1]
            
            if within_vars and np.mean(within_vars) > 0:
                separation_ratio = between_var / np.mean(within_vars)
                print(f"  → Separation ratio: {separation_ratio:.3f} (higher = better separation)")


if __name__ == "__main__":
    # Analyze the actual dataset
    df_metrics = analyze_real_dataset("test2")
    
    if df_metrics is not None:
        create_visualizations(df_metrics, "test2")
        analyze_quality_separation(df_metrics)
        
        print(f"\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        print("Generated plots for real data:")
        print("  - test2_skewness_vs_jaccard.png")
        print("  - test2_quality_clustering.png") 
        print("  - test2_pairwise_metrics.png")
    else:
        print("Unable to analyze dataset. Check data availability.")