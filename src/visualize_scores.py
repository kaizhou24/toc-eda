"""
visualize_scores.py

Visualization functions for circuit score data with symmetric log transformations.
Handles extreme values in distribution data using logarithmic scaling.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from .calculate_scores import (
    extract_edge_scores, 
    calculate_skewness, 
    calculate_kurtosis,
    calculate_basic_stats,
    calculate_average_jaccard_similarity
)


def symmetric_log_transform(x: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """Apply symmetric logarithmic transformation: sign(x) * log(1 + |x|).
    
    Args:
        x: Input array of values
        threshold: Linear threshold around zero (values below this remain linear)
        
    Returns:
        Transformed array with compressed extreme values
    """
    return np.sign(x) * np.log(1 + np.abs(x) / threshold)


def inverse_symmetric_log_transform(x: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """Inverse of symmetric logarithmic transformation.
    
    Args:
        x: Transformed array of values
        threshold: Linear threshold used in original transformation
        
    Returns:
        Original scale array
    """
    return threshold * np.sign(x) * (np.exp(np.abs(x)) - 1)


def plot_distribution_comparison(scores: List[float], title: str = "Score Distribution", 
                                figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """Plot comparison of raw vs log-transformed distributions.
    
    Args:
        scores: List of score values
        title: Plot title
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    if not scores:
        raise ValueError("Cannot plot empty scores list")
        
    scores_array = np.array(scores)
    log_scores = symmetric_log_transform(scores_array)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Raw distribution
    axes[0].hist(scores_array, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_title(f'{title} - Raw Data')
    axes[0].set_xlabel('Score Value')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Log-transformed distribution
    axes[1].hist(log_scores, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1].set_title(f'{title} - Log Transformed')
    axes[1].set_xlabel('Log(Score Value)')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    
    # Q-Q plot for normality assessment
    stats.probplot(log_scores, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot (Log Transformed)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_box_comparison(scores: List[float], title: str = "Score Distribution",
                       figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """Plot box plots comparing raw vs log-transformed distributions.
    
    Args:
        scores: List of score values
        title: Plot title
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    if not scores:
        raise ValueError("Cannot plot empty scores list")
        
    scores_array = np.array(scores)
    log_scores = symmetric_log_transform(scores_array)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Raw box plot
    axes[0].boxplot(scores_array, vert=True)
    axes[0].set_title(f'{title} - Raw Data')
    axes[0].set_ylabel('Score Value')
    axes[0].grid(True, alpha=0.3)
    
    # Log-transformed box plot
    axes[1].boxplot(log_scores, vert=True)
    axes[1].set_title(f'{title} - Log Transformed')
    axes[1].set_ylabel('Log(Score Value)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_scatter_with_transform(x_scores: List[float], y_scores: List[float],
                               x_label: str = "X Scores", y_label: str = "Y Scores",
                               title: str = "Score Relationship",
                               figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """Plot scatter plots comparing raw vs log-transformed relationships.
    
    Args:
        x_scores: List of x-axis score values
        y_scores: List of y-axis score values
        x_label: Label for x-axis
        y_label: Label for y-axis
        title: Plot title
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    if len(x_scores) != len(y_scores):
        raise ValueError("x_scores and y_scores must have same length")
    if not x_scores or not y_scores:
        raise ValueError("Cannot plot empty scores lists")
        
    x_array = np.array(x_scores)
    y_array = np.array(y_scores)
    x_log = symmetric_log_transform(x_array)
    y_log = symmetric_log_transform(y_array)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Raw scatter
    axes[0].scatter(x_array, y_array, alpha=0.6, s=20)
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].set_title(f'{title} - Raw Data')
    axes[0].grid(True, alpha=0.3)
    
    # Log-transformed scatter
    axes[1].scatter(x_log, y_log, alpha=0.6, s=20, color='red')
    axes[1].set_xlabel(f'Log({x_label})')
    axes[1].set_ylabel(f'Log({y_label})')
    axes[1].set_title(f'{title} - Log Transformed')
    axes[1].grid(True, alpha=0.3)
    
    # Correlation comparison
    raw_corr = np.corrcoef(x_array, y_array)[0, 1]
    log_corr = np.corrcoef(x_log, y_log)[0, 1]
    
    axes[2].bar(['Raw Data', 'Log Transformed'], [raw_corr, log_corr], 
               color=['skyblue', 'lightcoral'], alpha=0.7)
    axes[2].set_ylabel('Correlation Coefficient')
    axes[2].set_title('Correlation Comparison')
    axes[2].set_ylim(-1, 1)
    axes[2].grid(True, alpha=0.3)
    
    # Add correlation values as text
    axes[2].text(0, raw_corr + 0.05, f'{raw_corr:.3f}', ha='center', fontweight='bold')
    axes[2].text(1, log_corr + 0.05, f'{log_corr:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    return fig


def visualize_circuit_scores(circuit_data: Dict[str, Any], title: str = "Circuit Scores") -> plt.Figure:
    """Visualize scores from a single circuit with log transformation.
    
    Args:
        circuit_data: Circuit dictionary with edge score data
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    scores = extract_edge_scores(circuit_data)
    if not scores:
        raise ValueError("No valid scores found in circuit data")
        
    return plot_distribution_comparison(scores, title)


def visualize_multiple_circuits(circuits_list: List[Dict[str, Any]], 
                               labels: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """Visualize and compare scores from multiple circuits.
    
    Args:
        circuits_list: List of circuit dictionaries
        labels: Optional labels for each circuit
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    if not circuits_list:
        raise ValueError("Cannot visualize empty circuits list")
        
    n_circuits = len(circuits_list)
    if labels is None:
        labels = [f"Circuit {i+1}" for i in range(n_circuits)]
    
    fig, axes = plt.subplots(2, n_circuits, figsize=figsize)
    if n_circuits == 1:
        axes = axes.reshape(-1, 1)
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_circuits))
    
    for i, (circuit_data, label, color) in enumerate(zip(circuits_list, labels, colors)):
        scores = extract_edge_scores(circuit_data)
        if not scores:
            continue
            
        scores_array = np.array(scores)
        log_scores = symmetric_log_transform(scores_array)
        
        # Raw distribution
        axes[0, i].hist(scores_array, bins=30, alpha=0.7, color=color, edgecolor='black')
        axes[0, i].set_title(f'{label} - Raw')
        axes[0, i].set_xlabel('Score Value')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].grid(True, alpha=0.3)
        
        # Log-transformed distribution
        axes[1, i].hist(log_scores, bins=30, alpha=0.7, color=color, edgecolor='black')
        axes[1, i].set_title(f'{label} - Log Transformed')
        axes[1, i].set_xlabel('Log(Score Value)')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_summary_statistics_table(circuits_list: List[Dict[str, Any]], 
                                   labels: Optional[List[str]] = None) -> pd.DataFrame:
    """Create summary statistics table comparing raw vs log-transformed data.
    
    Args:
        circuits_list: List of circuit dictionaries
        labels: Optional labels for each circuit
        
    Returns:
        DataFrame with summary statistics
    """
    if not circuits_list:
        raise ValueError("Cannot create summary for empty circuits list")
        
    if labels is None:
        labels = [f"Circuit {i+1}" for i in range(len(circuits_list))]
    
    summary_data = []
    
    for circuit_data, label in zip(circuits_list, labels):
        scores = extract_edge_scores(circuit_data)
        if not scores:
            continue
            
        scores_array = np.array(scores)
        log_scores = symmetric_log_transform(scores_array)
        
        # Raw statistics
        raw_stats = {
            'Circuit': label,
            'Transform': 'Raw',
            'Count': len(scores_array),
            'Mean': np.mean(scores_array),
            'Std': np.std(scores_array),
            'Min': np.min(scores_array),
            'Max': np.max(scores_array),
            'Skewness': stats.skew(scores_array),
            'Kurtosis': stats.kurtosis(scores_array)
        }
        summary_data.append(raw_stats)
        
        # Log-transformed statistics
        log_stats = {
            'Circuit': label,
            'Transform': 'Log',
            'Count': len(log_scores),
            'Mean': np.mean(log_scores),
            'Std': np.std(log_scores),
            'Min': np.min(log_scores),
            'Max': np.max(log_scores),
            'Skewness': stats.skew(log_scores),
            'Kurtosis': stats.kurtosis(log_scores)
        }
        summary_data.append(log_stats)
    
    return pd.DataFrame(summary_data)


def calculate_metrics_for_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all metrics for each circuit in the dataset.
    
    Args:
        df: DataFrame with circuit_data and quality_label columns
        
    Returns:
        DataFrame with calculated metrics for each circuit
    """
    metrics_data = []
    
    # Get all circuits with data for Jaccard similarity calculations
    valid_circuits = df[df['circuit_data'].notna()]['circuit_data'].tolist()
    
    for idx, row in df.iterrows():
        if pd.isna(row['circuit_data']):
            continue
            
        circuit_data = row['circuit_data']
        
        # Calculate basic statistics
        basic_stats = calculate_basic_stats(circuit_data)
        
        # Calculate average Jaccard similarity (excluding self)
        other_circuits = [c for i, c in enumerate(valid_circuits) if i != idx]
        avg_jaccard = calculate_average_jaccard_similarity(circuit_data, other_circuits, exclude_self=True)
        
        # Combine all metrics
        metrics = {
            'thought_index': row.get('thought_index', idx),
            'quality_label': row.get('quality_label', 0),
            'avg_jaccard_similarity': avg_jaccard,
            **basic_stats
        }
        
        metrics_data.append(metrics)
    
    return pd.DataFrame(metrics_data)


def plot_metric_relationship_with_labels(df_metrics: pd.DataFrame, 
                                        x_metric: str, y_metric: str,
                                        apply_log_transform: bool = True,
                                        color_by_quality: bool = True,
                                        figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """Plot relationship between two metrics with quality labels.
    
    Args:
        df_metrics: DataFrame with calculated metrics and quality_label column
        x_metric: Name of metric for x-axis
        y_metric: Name of metric for y-axis
        apply_log_transform: Whether to apply symmetric log transformation
        color_by_quality: Whether to color points by quality label
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    # Filter out rows with missing values for the selected metrics
    valid_data = df_metrics.dropna(subset=[x_metric, y_metric])
    
    if valid_data.empty:
        raise ValueError(f"No valid data found for metrics {x_metric} and {y_metric}")
    
    x_values = valid_data[x_metric].values
    y_values = valid_data[y_metric].values
    
    if apply_log_transform:
        x_transformed = symmetric_log_transform(x_values)
        y_transformed = symmetric_log_transform(y_values)
        x_label = f"Log({x_metric})"
        y_label = f"Log({y_metric})"
    else:
        x_transformed = x_values
        y_transformed = y_values
        x_label = x_metric
        y_label = y_metric
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    if color_by_quality:
        quality_labels = valid_data['quality_label'].values
        unique_labels = sorted(valid_data['quality_label'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        
        # Scatter plot colored by quality
        for i, label in enumerate(unique_labels):
            mask = quality_labels == label
            axes[0].scatter(x_transformed[mask], y_transformed[mask], 
                           c=[colors[i]], label=f'Quality {label}', 
                           alpha=0.7, s=50)
        
        axes[0].legend()
        axes[0].set_title(f'{y_label} vs {x_label} (Colored by Quality)')
    else:
        axes[0].scatter(x_transformed, y_transformed, alpha=0.7, s=50)
        axes[0].set_title(f'{y_label} vs {x_label}')
    
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].grid(True, alpha=0.3)
    
    # Box plot showing distribution by quality label
    if color_by_quality:
        quality_labels = valid_data['quality_label'].values
        box_data = []
        box_labels = []
        
        for label in unique_labels:
            mask = quality_labels == label
            if apply_log_transform:
                box_data.append(y_transformed[mask])
            else:
                box_data.append(y_values[mask])
            box_labels.append(f'Q{label}')
        
        axes[1].boxplot(box_data, labels=box_labels)
        axes[1].set_title(f'{y_label} Distribution by Quality')
        axes[1].set_ylabel(y_label)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].hist(y_transformed if apply_log_transform else y_values, 
                     bins=20, alpha=0.7, edgecolor='black')
        axes[1].set_title(f'{y_label} Distribution')
        axes[1].set_xlabel(y_label)
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_quality_clustering_analysis(df_metrics: pd.DataFrame, 
                                    metrics_to_plot: List[str] = None,
                                    apply_log_transform: bool = True,
                                    figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
    """Create comprehensive quality clustering analysis plots.
    
    Args:
        df_metrics: DataFrame with calculated metrics and quality_label column
        metrics_to_plot: List of metric names to include. If None, uses key metrics
        apply_log_transform: Whether to apply symmetric log transformation
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['skewness', 'avg_jaccard_similarity', 'kurtosis', 'std']
    
    # Filter available metrics
    available_metrics = [m for m in metrics_to_plot if m in df_metrics.columns]
    if len(available_metrics) < 2:
        raise ValueError("Need at least 2 valid metrics for clustering analysis")
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=figsize)
    axes = axes.flatten()
    
    unique_labels = sorted(df_metrics['quality_label'].unique())
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, metric in enumerate(available_metrics):
        valid_data = df_metrics.dropna(subset=[metric])
        
        if valid_data.empty:
            continue
            
        metric_values = valid_data[metric].values
        if apply_log_transform:
            metric_values = symmetric_log_transform(metric_values)
            metric_label = f"Log({metric})"
        else:
            metric_label = metric
        
        # Create box plots grouped by quality
        quality_groups = []
        group_labels = []
        
        for j, label in enumerate(unique_labels):
            group_data = valid_data[valid_data['quality_label'] == label]
            if not group_data.empty:
                group_values = group_data[metric].values
                if apply_log_transform:
                    group_values = symmetric_log_transform(group_values)
                quality_groups.append(group_values)
                group_labels.append(f'Q{label}')
        
        if quality_groups:
            bp = axes[i].boxplot(quality_groups, labels=group_labels, patch_artist=True)
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], colors[:len(quality_groups)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        axes[i].set_title(f'{metric_label} by Quality')
        axes[i].set_ylabel(metric_label)
        axes[i].grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(available_metrics), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Quality Label Clustering Analysis', fontsize=16, y=0.98)
    plt.tight_layout()
    return fig


def plot_pairwise_metric_relationships(df_metrics: pd.DataFrame,
                                     metrics_to_plot: List[str] = None,
                                     apply_log_transform: bool = True,
                                     figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """Create pairwise scatter plots of metrics colored by quality labels.
    
    Args:
        df_metrics: DataFrame with calculated metrics and quality_label column
        metrics_to_plot: List of metric names to include. If None, uses key metrics
        apply_log_transform: Whether to apply symmetric log transformation
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['skewness', 'avg_jaccard_similarity', 'kurtosis', 'mean']
    
    # Filter available metrics and remove rows with any missing values
    available_metrics = [m for m in metrics_to_plot if m in df_metrics.columns]
    clean_data = df_metrics.dropna(subset=available_metrics)
    
    if len(available_metrics) < 2:
        raise ValueError("Need at least 2 valid metrics for pairwise analysis")
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(n_metrics, n_metrics, figsize=figsize)
    
    unique_labels = sorted(clean_data['quality_label'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    for i, metric_y in enumerate(available_metrics):
        for j, metric_x in enumerate(available_metrics):
            if i == j:
                # Diagonal: histogram of the metric
                metric_values = clean_data[metric_x].values
                if apply_log_transform:
                    metric_values = symmetric_log_transform(metric_values)
                
                axes[i, j].hist(metric_values, bins=20, alpha=0.7, edgecolor='black')
                axes[i, j].set_title(f'{metric_x}')
            else:
                # Off-diagonal: scatter plot
                x_values = clean_data[metric_x].values
                y_values = clean_data[metric_y].values
                
                if apply_log_transform:
                    x_values = symmetric_log_transform(x_values)
                    y_values = symmetric_log_transform(y_values)
                
                # Plot each quality label with different colors
                for k, label in enumerate(unique_labels):
                    mask = clean_data['quality_label'].values == label
                    axes[i, j].scatter(x_values[mask], y_values[mask], 
                                     c=[colors[k]], label=f'Q{label}', 
                                     alpha=0.7, s=30)
                
                if i == n_metrics - 1:  # Bottom row
                    axes[i, j].set_xlabel(f'{"Log(" + metric_x + ")" if apply_log_transform else metric_x}')
                if j == 0:  # Left column
                    axes[i, j].set_ylabel(f'{"Log(" + metric_y + ")" if apply_log_transform else metric_y}')
                
                # Add legend to top-right plot
                if i == 0 and j == n_metrics - 1:
                    axes[i, j].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            axes[i, j].grid(True, alpha=0.3)
    
    plt.suptitle('Pairwise Metric Relationships by Quality Label', fontsize=16, y=0.98)
    plt.tight_layout()
    return fig