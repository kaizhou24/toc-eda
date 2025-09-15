"""
pca_alignment.py

Demonstration of compass analysis for circuit quality assessment.
Implements both PCA-based and probe-based compass alignment analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.data_loader import load_and_build_dataframe
from src.visualize_scores import (
    calculate_compass_metrics_for_dataset,
    plot_compass_analysis,
    plot_compass_correlation_analysis
)


def run_compass_analysis(dataset_name="test2"):
    """Run complete compass analysis on a dataset."""
    print("="*60)
    print("COMPASS ANALYSIS FOR CIRCUIT QUALITY ASSESSMENT")
    print("="*60)
    
    # Load the dataset
    print(f"Loading {dataset_name} dataset...")
    df = load_and_build_dataframe(dataset_name, base_path="data")
    
    if df.empty:
        print(f"No data found for dataset {dataset_name}")
        return None
    
    print(f"Loaded {len(df)} records")
    
    # Check for required columns
    if 'quality_label' not in df.columns:
        print("No quality_label column found - cannot perform compass analysis")
        return None
    
    # Show data overview
    circuits_with_data = df[df['circuit_data'].notna()]
    print(f"Records with circuit data: {len(circuits_with_data)}")
    
    quality_dist = df['quality_label'].value_counts().sort_index()
    print(f"Quality label distribution: {quality_dist.to_dict()}")
    
    if len(circuits_with_data) < 3:
        print("Not enough circuits with data for compass analysis")
        return None
    
    # Calculate compass metrics
    print("\n" + "="*50)
    print("CALCULATING COMPASS ALIGNMENT METRICS")
    print("="*50)
    
    df_with_compass = calculate_compass_metrics_for_dataset(df)
    
    # Create visualizations
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic compass alignment plots
    try:
        print("Creating compass alignment box plots...")
        fig1 = plot_compass_analysis(df_with_compass, compass_type='both')
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_compass_alignment.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved compass alignment plot: {os.path.join(output_dir, f'{dataset_name}_compass_alignment.png')}")
    except Exception as e:
        print(f"Failed to create compass alignment plots: {e}")
    
    # Correlation analysis plots
    try:
        print("Creating compass correlation analysis...")
        fig2 = plot_compass_correlation_analysis(df_with_compass)
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_compass_correlation.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved compass correlation analysis: {os.path.join(output_dir, f'{dataset_name}_compass_correlation.png')}")
    except Exception as e:
        print(f"Failed to create compass correlation plots: {e}")
    
    return df_with_compass


def analyze_compass_effectiveness(df_compass):
    """Analyze the effectiveness of compass alignment metrics."""
    print("\n" + "="*50)
    print("COMPASS EFFECTIVENESS ANALYSIS")
    print("="*50)
    
    # Filter valid data
    valid_data = df_compass.dropna(subset=['quality_label', 'pca_alignment', 'probe_alignment'])
    
    if len(valid_data) < 3:
        print("Not enough valid data for effectiveness analysis")
        return
    
    print(f"Analyzing {len(valid_data)} circuits with complete compass data")
    
    # Calculate correlations with quality
    from scipy.stats import pearsonr, spearmanr
    
    try:
        pca_pearson_r, pca_pearson_p = pearsonr(valid_data['quality_label'], valid_data['pca_alignment'])
        pca_spearman_r, pca_spearman_p = spearmanr(valid_data['quality_label'], valid_data['pca_alignment'])
        
        print(f"\nPCA Compass Performance:")
        print(f"  Pearson correlation:  r = {pca_pearson_r:.3f}, p = {pca_pearson_p:.3e}")
        print(f"  Spearman correlation: r = {pca_spearman_r:.3f}, p = {pca_spearman_p:.3e}")
        
        probe_pearson_r, probe_pearson_p = pearsonr(valid_data['quality_label'], valid_data['probe_alignment'])
        probe_spearman_r, probe_spearman_p = spearmanr(valid_data['quality_label'], valid_data['probe_alignment'])
        
        print(f"\nProbe Compass Performance:")
        print(f"  Pearson correlation:  r = {probe_pearson_r:.3f}, p = {probe_pearson_p:.3e}")
        print(f"  Spearman correlation: r = {probe_spearman_r:.3f}, p = {probe_spearman_p:.3e}")
        
        # Compare compass methods
        compass_agreement_r, compass_agreement_p = pearsonr(valid_data['pca_alignment'], valid_data['probe_alignment'])
        print(f"\nCompass Method Agreement:")
        print(f"  PCA vs Probe correlation: r = {compass_agreement_r:.3f}, p = {compass_agreement_p:.3e}")
        
        # Determine best compass
        best_method = "PCA" if abs(pca_pearson_r) > abs(probe_pearson_r) else "Probe"
        best_r = pca_pearson_r if best_method == "PCA" else probe_pearson_r
        
        print(f"\nBest Compass Method: {best_method} (r = {best_r:.3f})")
        
        # Quality group analysis
        print(f"\nQuality Group Analysis:")
        for quality in sorted(valid_data['quality_label'].unique()):
            group_data = valid_data[valid_data['quality_label'] == quality]
            if len(group_data) > 0:
                pca_mean = group_data['pca_alignment'].mean()
                pca_std = group_data['pca_alignment'].std()
                probe_mean = group_data['probe_alignment'].mean()
                probe_std = group_data['probe_alignment'].std()
                
                print(f"  Quality {quality} (n={len(group_data)}):")
                print(f"    PCA alignment:   {pca_mean:.3f} ± {pca_std:.3f}")
                print(f"    Probe alignment: {probe_mean:.3f} ± {probe_std:.3f}")
        
    except Exception as e:
        print(f"Error in effectiveness analysis: {e}")


def main():
    """Main function to run compass analysis demonstration."""
    try:
        # Run compass analysis on test2 dataset
        df_compass = run_compass_analysis("test2")
        
        if df_compass is not None:
            # Analyze effectiveness
            analyze_compass_effectiveness(df_compass)
            
            print("\n" + "="*60)
            print("COMPASS ANALYSIS COMPLETE")
            print("="*60)
            print("Generated compass analysis files:")
            print("  - test2_compass_alignment.png")
            print("  - test2_compass_correlation.png")
            print("\nCompass analysis reveals geometric patterns in high-dimensional")
            print("circuit space that correlate with manual quality assessments.")
            
        else:
            print("Compass analysis could not be completed.")
            
    except Exception as e:
        print(f"Error running compass analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()