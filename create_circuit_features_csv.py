#!/usr/bin/env python3
"""
Clean version of circuit features CSV export functionality.
"""

import sys
import os
from pathlib import Path
sys.path.append('src')

import pandas as pd
import numpy as np
from data_loader import load_and_build_dataframe
from calculate_scores import (
    calculate_basic_stats, 
    calculate_circuit_aggregates,
    calculate_percentiles,
    calculate_pca_compass_alignment,
    calculate_probe_compass_alignment
)

def create_circuit_features_csv(dataset_name="test2", output_file=None):
    """Create a comprehensive CSV with circuit features including compass alignment.
    
    Args:
        dataset_name: Name of dataset to process
        output_file: Output CSV filename (auto-generated if None)
    
    Returns:
        Path to generated CSV file
    """
    print(f"Loading {dataset_name} dataset...")
    df = load_and_build_dataframe(dataset_name, base_path="data")
    
    if df.empty:
        raise ValueError("No data found")
    
    circuits_df = df[df['circuit_data'].notna()].copy()
    print(f"Processing {len(circuits_df)} circuits")
    
    # Extract all circuit data and quality labels
    all_circuits = circuits_df['circuit_data'].tolist()
    quality_labels = circuits_df['quality_label'].tolist()
    
    # Calculate compass alignments
    compass_data = {}
    
    if len(all_circuits) >= 3:
        print("Calculating compass alignments...")
        
        pca_result = calculate_pca_compass_alignment(all_circuits, quality_labels)
        if pca_result:
            compass_data['pca_alignment'] = pca_result['alignment_scores']
            compass_data['pca_explained_variance'] = pca_result['explained_variance_ratio']
        
        probe_result = calculate_probe_compass_alignment(all_circuits, quality_labels)
        if probe_result:
            compass_data['probe_alignment'] = probe_result['alignment_scores']
            compass_data['probe_training_accuracy'] = probe_result['training_accuracy']
    
    # Calculate individual circuit features
    print("Calculating individual circuit features...")
    feature_rows = []
    
    for i, (idx, row) in enumerate(circuits_df.iterrows()):
        circuit_data = row['circuit_data']
        
        # Initialize feature dictionary
        features = {
            'circuit_id': idx,
            'thought_index': row.get('thought_index', idx),
            'quality_label': row.get('quality_label', 0)
        }
        
        # Add compass features
        for compass_feature, values in compass_data.items():
            if compass_feature.endswith('_accuracy') or compass_feature.endswith('_variance'):
                # These are constant values for all circuits
                features[compass_feature] = values
            else:
                # These are per-circuit values
                features[compass_feature] = values[i] if i < len(values) else None
        
        # Calculate per-circuit features
        basic_stats = calculate_basic_stats(circuit_data)
        if basic_stats:
            features.update(basic_stats)
        
        aggregates = calculate_circuit_aggregates(circuit_data)
        if aggregates:
            features.update(aggregates)
        
        percentiles = calculate_percentiles(circuit_data)
        if percentiles:
            features.update(percentiles)
        
        feature_rows.append(features)
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(circuits_df)} circuits")
    
    # Create DataFrame
    df_features = pd.DataFrame(feature_rows)
    
    # Remove any duplicate columns
    df_features = df_features.loc[:, ~df_features.columns.duplicated()]
    
    # Reorder columns for clarity
    metadata_cols = ['circuit_id', 'thought_index', 'quality_label']
    compass_cols = [col for col in df_features.columns 
                   if 'alignment' in col or col.startswith('pca_') or col.startswith('probe_')]
    basic_cols = ['mean', 'std', 'min', 'max', 'median', 'skewness', 'kurtosis']
    aggregate_cols = [col for col in df_features.columns if col.startswith('circuit_')]
    percentile_cols = [col for col in df_features.columns if col.startswith('p') and col[1:].isdigit()]
    
    # Build final column order
    column_order = []
    for col_group in [metadata_cols, compass_cols, basic_cols, aggregate_cols, percentile_cols]:
        column_order.extend([col for col in col_group if col in df_features.columns])
    
    # Add any remaining columns
    remaining_cols = [col for col in df_features.columns if col not in column_order]
    column_order.extend(remaining_cols)
    
    df_features = df_features[column_order]
    
    # Generate output filename
    if output_file is None:
        # Ensure data/processed directory exists
        os.makedirs("data/processed", exist_ok=True)
        output_file = f"data/processed/{dataset_name}_circuit_features.csv"
    
    # Ensure output directory exists for custom paths
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export to CSV
    df_features.to_csv(output_file, index=False, float_format='%.6f')
    
    print(f"\n✅ Successfully exported circuit features:")
    print(f"   File: {output_file}")
    print(f"   Shape: {df_features.shape} (rows × columns)")
    print(f"   Features: {len(df_features.columns) - 3}")  # Exclude metadata columns
    
    # Calculate and display correlations with quality
    print(f"\n📊 Top features correlated with quality:")
    
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['circuit_id', 'thought_index', 'quality_label']]
    
    correlations = []
    for col in feature_cols:
        valid_data = df_features[[col, 'quality_label']].dropna()
        if len(valid_data) > 2:
            corr_matrix = valid_data.corr()
            corr_value = corr_matrix.loc[col, 'quality_label']
            if not np.isnan(corr_value):
                correlations.append((col, corr_value))
    
    # Sort by absolute correlation value
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for i, (feature, corr) in enumerate(correlations[:10], 1):
        print(f"   {i:2d}. {feature:25s}: r = {corr:6.3f}")
    
    # Show sample of the data
    print(f"\n📋 Sample data (first 3 rows, key columns):")
    sample_cols = ['circuit_id', 'quality_label']
    if 'probe_alignment' in df_features.columns:
        sample_cols.append('probe_alignment')
    sample_cols.extend(['mean', 'std', 'circuit_max'])
    
    available_cols = [col for col in sample_cols if col in df_features.columns]
    sample_data = df_features[available_cols].head(3).round(3)
    
    print(sample_data.to_string(index=False))
    
    return output_file

def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export circuit features to CSV')
    parser.add_argument('dataset', nargs='?', default='test2', help='Dataset name (default: test2)')
    parser.add_argument('-o', '--output', help='Output CSV file path')
    
    args = parser.parse_args()
    
    try:
        output_file = create_circuit_features_csv(args.dataset, args.output)
        print(f"\n🎉 Export complete! Generated: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())