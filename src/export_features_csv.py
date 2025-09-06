"""
export_features_csv.py

Functions to extract all circuit features and export them to CSV format using pandas.
Combines individual circuit metrics with compass alignment scores for comprehensive analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

from data_loader import load_and_build_dataframe
from calculate_scores import (
    calculate_basic_stats,
    calculate_skewness, 
    calculate_kurtosis,
    calculate_percentiles,
    calculate_average_jaccard_similarity,
    calculate_circuit_aggregates,
    calculate_pca_compass_alignment,
    calculate_probe_compass_alignment,
    calculate_circuit_cka_similarity,
    calculate_circuit_procrustes_disparity,
    statistical_significance_tests
)


def calculate_all_circuit_features(circuit_data: Dict[str, Any], 
                                 other_circuits: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Calculate all available features for a single circuit.
    
    Args:
        circuit_data: Single circuit dictionary with edge score data
        other_circuits: List of other circuits for similarity calculations
        
    Returns:
        Dictionary with all calculated features
    """
    features = {}
    
    # Basic statistical features
    basic_stats = calculate_basic_stats(circuit_data)
    features.update(basic_stats)
    
    # Individual statistical measures
    features['skewness_individual'] = calculate_skewness(circuit_data)
    features['kurtosis_individual'] = calculate_kurtosis(circuit_data)
    
    # Percentiles
    percentiles = calculate_percentiles(circuit_data)
    features.update(percentiles)
    
    # Circuit aggregates
    aggregates = calculate_circuit_aggregates(circuit_data)
    features.update(aggregates)
    
    # Average Jaccard similarity (if other circuits provided)
    if other_circuits:
        avg_jaccard = calculate_average_jaccard_similarity(circuit_data, other_circuits)
        features['avg_jaccard_similarity'] = avg_jaccard
    else:
        features['avg_jaccard_similarity'] = None
    
    return features


def extract_dataset_features(dataset_name: str, 
                           base_path: str = "data",
                           include_compass: bool = True) -> pd.DataFrame:
    """Extract all features from a dataset and return as pandas DataFrame.
    
    Args:
        dataset_name: Name of the dataset to process
        base_path: Base directory path for data files
        include_compass: Whether to include compass alignment features
        
    Returns:
        pandas DataFrame with all circuit features
    """
    # Load the dataset
    df = load_and_build_dataframe(dataset_name, base_path=base_path)
    
    if df.empty:
        raise ValueError(f"No data found for dataset {dataset_name}")
    
    # Filter circuits with valid data
    circuits_df = df[df['circuit_data'].notna()].copy()
    if len(circuits_df) == 0:
        raise ValueError(f"No circuits with valid data found in {dataset_name}")
    
    print(f"Processing {len(circuits_df)} circuits from {dataset_name}")
    
    # Extract all circuit data for similarity calculations
    all_circuits = circuits_df['circuit_data'].tolist()
    quality_labels = circuits_df['quality_label'].tolist()
    
    # Calculate compass alignments if requested
    compass_features = {}
    if include_compass and len(all_circuits) >= 3:
        print("Calculating PCA compass alignment...")
        pca_result = calculate_pca_compass_alignment(all_circuits, quality_labels)
        
        print("Calculating probe compass alignment...")
        probe_result = calculate_probe_compass_alignment(all_circuits, quality_labels)
        
        if pca_result:
            compass_features['pca_alignment'] = pca_result['alignment_scores']
            compass_features['pca_explained_variance'] = [pca_result['explained_variance_ratio']] * len(all_circuits)
        
        if probe_result:
            compass_features['probe_alignment'] = probe_result['alignment_scores']
            compass_features['probe_training_accuracy'] = [probe_result['training_accuracy']] * len(all_circuits)
        
        # Skip expensive CKA and Procrustes calculations for now
        # These can be added back if needed for specific analysis
        print("Skipping CKA and Procrustes calculations (too expensive for routine export)")
    
    # Calculate individual circuit features
    print("Calculating individual circuit features...")
    feature_rows = []
    
    for idx, row in circuits_df.iterrows():
        circuit_data = row['circuit_data']
        
        # Basic features
        circuit_features = calculate_all_circuit_features(circuit_data, all_circuits)
        
        # Add metadata
        circuit_features['circuit_id'] = idx
        circuit_features['thought_index'] = row.get('thought_index', idx)
        circuit_features['quality_label'] = row.get('quality_label', 0)
        
        # Add compass features for this circuit
        circuit_idx = circuits_df.index.get_loc(idx)
        for feature_name, feature_values in compass_features.items():
            if isinstance(feature_values, list) and circuit_idx < len(feature_values):
                circuit_features[feature_name] = feature_values[circuit_idx]
            else:
                circuit_features[feature_name] = None
        
        feature_rows.append(circuit_features)
    
    # Create DataFrame
    df_features = pd.DataFrame(feature_rows)
    
    # Reorder columns for better readability
    metadata_cols = ['circuit_id', 'thought_index', 'quality_label']
    compass_cols = [col for col in df_features.columns if 'alignment' in col or 'cka' in col or 'procrustes' in col or 'pca_' in col or 'probe_' in col]
    basic_stats_cols = ['mean', 'std', 'min', 'max', 'median', 'skewness', 'kurtosis']
    aggregate_cols = [col for col in df_features.columns if col.startswith('circuit_')]
    percentile_cols = [col for col in df_features.columns if col.startswith('p')]
    other_cols = [col for col in df_features.columns if col not in metadata_cols + compass_cols + basic_stats_cols + aggregate_cols + percentile_cols]
    
    column_order = metadata_cols + compass_cols + basic_stats_cols + aggregate_cols + percentile_cols + other_cols
    column_order = [col for col in column_order if col in df_features.columns]
    
    df_features = df_features[column_order]
    
    print(f"Generated feature dataset with {len(df_features)} rows and {len(df_features.columns)} columns")
    
    return df_features


def export_features_to_csv(dataset_name: str,
                          output_path: Optional[str] = None,
                          base_path: str = "data",
                          include_compass: bool = True) -> str:
    """Extract circuit features and export to CSV file.
    
    Args:
        dataset_name: Name of the dataset to process
        output_path: Output CSV file path (if None, auto-generates to data/processed/)
        base_path: Base directory path for data files
        include_compass: Whether to include compass alignment features
        
    Returns:
        Path to the generated CSV file
    """
    # Extract features
    df_features = extract_dataset_features(dataset_name, base_path, include_compass)
    
    # Generate output path if not provided
    if output_path is None:
        # Default to data/processed directory
        output_path = f"data/processed/{dataset_name}_circuit_features.csv"
    
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Export to CSV
    df_features.to_csv(output_file, index=False, float_format='%.6f')
    
    print(f"✅ Exported {len(df_features)} circuit features to: {output_file}")
    print(f"📊 Dataset shape: {df_features.shape}")
    print(f"📋 Columns: {list(df_features.columns)}")
    
    return str(output_file)


def generate_feature_summary(df_features: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics for the feature dataset.
    
    Args:
        df_features: DataFrame with circuit features
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'dataset_shape': df_features.shape,
        'num_circuits': len(df_features),
        'num_features': len(df_features.columns) - 3,  # Exclude metadata columns
        'quality_distribution': df_features['quality_label'].value_counts().to_dict(),
        'missing_values': df_features.isnull().sum().to_dict(),
        'feature_types': df_features.dtypes.to_dict()
    }
    
    # Correlation with quality (for numeric columns)
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'quality_label']
    
    correlations = {}
    for col in numeric_cols:
        if df_features[col].notna().sum() > 2:  # Need at least 3 valid values
            corr = df_features[['quality_label', col]].corr().iloc[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr
    
    summary['quality_correlations'] = correlations
    
    return summary


def main():
    """Main function to demonstrate CSV export functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export circuit features to CSV')
    parser.add_argument('dataset', help='Dataset name to process')
    parser.add_argument('-o', '--output', help='Output CSV file path')
    parser.add_argument('--base-path', default='data', help='Base path for data files')
    parser.add_argument('--no-compass', action='store_true', help='Skip compass alignment features')
    
    args = parser.parse_args()
    
    try:
        # Export features
        output_file = export_features_to_csv(
            args.dataset,
            args.output,
            args.base_path,
            include_compass=not args.no_compass
        )
        
        # Load and summarize
        df = pd.read_csv(output_file)
        summary = generate_feature_summary(df)
        
        print("\n" + "="*60)
        print("FEATURE DATASET SUMMARY")
        print("="*60)
        print(f"Dataset shape: {summary['dataset_shape']}")
        print(f"Number of circuits: {summary['num_circuits']}")
        print(f"Number of features: {summary['num_features']}")
        print(f"Quality distribution: {summary['quality_distribution']}")
        
        # Show top correlations with quality
        correlations = summary['quality_correlations']
        if correlations:
            sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            print(f"\nTop 10 features correlated with quality:")
            for i, (feature, corr) in enumerate(sorted_corr[:10]):
                print(f"  {i+1:2d}. {feature}: r = {corr:.3f}")
        
        print(f"\n✅ CSV export complete: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()