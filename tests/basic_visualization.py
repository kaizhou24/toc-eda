"""
test_visualizations.py

Test script to demonstrate visualization functions with sample data.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.visualize_scores import (
    symmetric_log_transform, 
    plot_distribution_comparison,
    plot_box_comparison,
    plot_scatter_with_transform,
    create_summary_statistics_table
)


def create_sample_circuit_data():
    """Create sample circuit data with extreme values for testing."""
    # Generate scores with extreme values and long tails
    np.random.seed(42)
    
    # Mix of different distributions to create extreme values
    normal_scores = np.random.normal(0, 1, 500)
    extreme_positive = np.random.exponential(10, 50)
    extreme_negative = -np.random.exponential(10, 50)
    outliers = np.concatenate([
        np.random.normal(100, 5, 10),  # Extreme positive outliers
        np.random.normal(-100, 5, 10)  # Extreme negative outliers
    ])
    
    all_scores = np.concatenate([normal_scores, extreme_positive, extreme_negative, outliers])
    
    # Create circuit data structure
    edges = {}
    for i, score in enumerate(all_scores):
        edges[f"edge_{i}"] = {"score": float(score)}
    
    return {"edges": edges}


def test_transformations():
    """Test the symmetric log transformation."""
    print("Testing symmetric log transformation...")
    
    # Test with extreme values
    test_values = np.array([-1000, -10, -1, 0, 1, 10, 1000])
    transformed = symmetric_log_transform(test_values)
    
    print("Original values:", test_values)
    print("Transformed values:", transformed)
    print("Range compression:")
    print(f"  Original range: [{test_values.min():.1f}, {test_values.max():.1f}]")
    print(f"  Transformed range: [{transformed.min():.3f}, {transformed.max():.3f}]")
    print()


def test_visualizations():
    """Test visualization functions with sample data."""
    print("Creating sample circuit data...")
    circuit_data = create_sample_circuit_data()
    
    # Extract scores for testing
    scores = []
    for edge_data in circuit_data['edges'].values():
        scores.append(edge_data['score'])
    
    print(f"Generated {len(scores)} scores with range [{min(scores):.2f}, {max(scores):.2f}]")
    
    # Test distribution comparison plot
    print("Creating distribution comparison plot...")
    fig1 = plot_distribution_comparison(scores, "Sample Circuit Scores")
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'distribution_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Test box plot comparison
    print("Creating box plot comparison...")
    fig2 = plot_box_comparison(scores, "Sample Circuit Scores")
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'box_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Test scatter plot with two different score sets
    scores2 = [s + np.random.normal(0, 5) for s in scores[:len(scores)//2]]  # Related but noisy
    scores1_subset = scores[:len(scores2)]
    
    print("Creating scatter plot comparison...")
    fig3 = plot_scatter_with_transform(scores1_subset, scores2, 
                                     "Original Scores", "Modified Scores",
                                     "Score Relationship")
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'scatter_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Test summary statistics
    print("Creating summary statistics...")
    circuits_list = [circuit_data]
    summary_df = create_summary_statistics_table(circuits_list, ["Sample Circuit"])
    print("\nSummary Statistics:")
    print(summary_df.round(4))
    
    print("\nVisualization test completed successfully!")
    print("Generated plots saved as PNG files.")


if __name__ == "__main__":
    test_transformations()
    test_visualizations()