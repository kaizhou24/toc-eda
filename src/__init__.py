"""
Tree of Circuits Analysis Package

Core modules for circuit data analysis and visualization.
"""

from .calculate_scores import (
    extract_edge_scores,
    calculate_skewness,
    calculate_kurtosis,
    calculate_basic_stats,
    calculate_similarity,
    calculate_jaccard_similarity,
    calculate_average_jaccard_similarity,
    calculate_score_distribution,
    calculate_percentiles
)

from .data_loader import (
    load_thought_labels,
    parse_circuit_filename,
    load_circuit_data,
    create_unified_dataset,
    load_and_build_dataframe
)

from .visualize_scores import (
    symmetric_log_transform,
    inverse_symmetric_log_transform,
    plot_distribution_comparison,
    plot_box_comparison,
    plot_scatter_with_transform,
    visualize_circuit_scores,
    visualize_multiple_circuits,
    create_summary_statistics_table,
    calculate_metrics_for_dataset,
    plot_metric_relationship_with_labels,
    plot_quality_clustering_analysis,
    plot_pairwise_metric_relationships
)

__version__ = "0.1.0"
__author__ = "Tree of Circuits Analysis Team"