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
        axes[i].set_xlabel('Quality')
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
                axes[i, j].set_xlabel(f'{"Log(" + metric_x + ")" if apply_log_transform else metric_x}')
                axes[i, j].set_ylabel('Frequency')
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
                
                # Add axis labels to all scatter plots
                axes[i, j].set_xlabel(f'{"Log(" + metric_x + ")" if apply_log_transform else metric_x}')
                axes[i, j].set_ylabel(f'{"Log(" + metric_y + ")" if apply_log_transform else metric_y}')
                
                # Add legend to top-right plot
                if i == 0 and j == n_metrics - 1:
                    axes[i, j].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            axes[i, j].grid(True, alpha=0.3)
    
    plt.suptitle('Pairwise Metric Relationships by Quality Label', fontsize=16, y=0.98)
    plt.tight_layout()
    return fig


def plot_compass_analysis(df_metrics: pd.DataFrame,
                         compass_type: str = 'both',
                         figsize: Tuple[int, int] = (16, 6)) -> plt.Figure:
    """Plot compass alignment analysis results.
    
    Args:
        df_metrics: DataFrame with quality_label and compass alignment columns
        compass_type: Type of compass to plot ('pca', 'probe', or 'both')
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    required_cols = ['quality_label']
    
    if compass_type == 'pca':
        required_cols.append('pca_alignment')
        n_plots = 1
    elif compass_type == 'probe':
        required_cols.append('probe_alignment')
        n_plots = 1
    else:  # both
        required_cols.extend(['pca_alignment', 'probe_alignment'])
        n_plots = 2
    
    # Check if required columns exist
    missing_cols = [col for col in required_cols if col not in df_metrics.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter valid data
    valid_data = df_metrics.dropna(subset=required_cols)
    if valid_data.empty:
        raise ValueError("No valid data found for compass analysis")
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    unique_labels = sorted(valid_data['quality_label'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    plot_idx = 0
    
    if compass_type in ['pca', 'both']:
        sns.boxplot(ax=axes[plot_idx], x='quality_label', y='pca_alignment', data=valid_data)
        axes[plot_idx].set_title('PCA Compass Alignment by Quality Label')
        axes[plot_idx].set_xlabel('Quality Label')
        axes[plot_idx].set_ylabel('PCA Alignment Score')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    if compass_type in ['probe', 'both']:
        sns.boxplot(ax=axes[plot_idx], x='quality_label', y='probe_alignment', data=valid_data)
        axes[plot_idx].set_title('Probe Compass Alignment by Quality Label')
        axes[plot_idx].set_xlabel('Quality Label')
        axes[plot_idx].set_ylabel('Probe Alignment Score')
        axes[plot_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_compass_correlation_analysis(df_metrics: pd.DataFrame,
                                    figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """Create comprehensive correlation analysis for compass alignments.
    
    Args:
        df_metrics: DataFrame with quality_label and compass alignment columns
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    required_cols = ['quality_label', 'pca_alignment', 'probe_alignment']
    missing_cols = [col for col in required_cols if col not in df_metrics.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    valid_data = df_metrics.dropna(subset=required_cols)
    if valid_data.empty:
        raise ValueError("No valid data found for compass correlation analysis")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Scatter plot: PCA vs Quality
    axes[0, 0].scatter(valid_data['quality_label'], valid_data['pca_alignment'], 
                      alpha=0.6, s=50, color='blue')
    axes[0, 0].set_xlabel('Quality Label')
    axes[0, 0].set_ylabel('PCA Alignment Score')
    axes[0, 0].set_title('PCA Alignment vs Quality Label')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Calculate and display correlation
    from scipy.stats import pearsonr, spearmanr
    pca_pearson_r, pca_pearson_p = pearsonr(valid_data['quality_label'], valid_data['pca_alignment'])
    axes[0, 0].text(0.05, 0.95, f'Pearson r = {pca_pearson_r:.3f}\np = {pca_pearson_p:.3e}', 
                   transform=axes[0, 0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Scatter plot: Probe vs Quality
    axes[0, 1].scatter(valid_data['quality_label'], valid_data['probe_alignment'], 
                      alpha=0.6, s=50, color='red')
    axes[0, 1].set_xlabel('Quality Label')
    axes[0, 1].set_ylabel('Probe Alignment Score')
    axes[0, 1].set_title('Probe Alignment vs Quality Label')
    axes[0, 1].grid(True, alpha=0.3)
    
    probe_pearson_r, probe_pearson_p = pearsonr(valid_data['quality_label'], valid_data['probe_alignment'])
    axes[0, 1].text(0.05, 0.95, f'Pearson r = {probe_pearson_r:.3f}\np = {probe_pearson_p:.3e}', 
                   transform=axes[0, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Scatter plot: PCA vs Probe alignment
    axes[1, 0].scatter(valid_data['pca_alignment'], valid_data['probe_alignment'], 
                      alpha=0.6, s=50, color='green')
    axes[1, 0].set_xlabel('PCA Alignment Score')
    axes[1, 0].set_ylabel('Probe Alignment Score')
    axes[1, 0].set_title('PCA vs Probe Alignment')
    axes[1, 0].grid(True, alpha=0.3)
    
    pca_probe_r, pca_probe_p = pearsonr(valid_data['pca_alignment'], valid_data['probe_alignment'])
    axes[1, 0].text(0.05, 0.95, f'Pearson r = {pca_probe_r:.3f}\np = {pca_probe_p:.3e}', 
                   transform=axes[1, 0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Summary statistics table
    axes[1, 1].axis('off')
    
    # Create summary table
    summary_data = [
        ['Metric', 'Pearson r', 'Pearson p', 'Spearman r', 'Spearman p'],
        ['PCA vs Quality', f'{pca_pearson_r:.3f}', f'{pca_pearson_p:.2e}', '', ''],
        ['Probe vs Quality', f'{probe_pearson_r:.3f}', f'{probe_pearson_p:.2e}', '', ''],
        ['PCA vs Probe', f'{pca_probe_r:.3f}', f'{pca_probe_p:.2e}', '', '']
    ]
    
    # Add Spearman correlations
    pca_spearman_r, pca_spearman_p = spearmanr(valid_data['quality_label'], valid_data['pca_alignment'])
    probe_spearman_r, probe_spearman_p = spearmanr(valid_data['quality_label'], valid_data['probe_alignment'])
    pca_probe_spearman_r, pca_probe_spearman_p = spearmanr(valid_data['pca_alignment'], valid_data['probe_alignment'])
    
    summary_data[1][3] = f'{pca_spearman_r:.3f}'
    summary_data[1][4] = f'{pca_spearman_p:.2e}'
    summary_data[2][3] = f'{probe_spearman_r:.3f}'
    summary_data[2][4] = f'{probe_spearman_p:.2e}'
    summary_data[3][3] = f'{pca_probe_spearman_r:.3f}'
    summary_data[3][4] = f'{pca_probe_spearman_p:.2e}'
    
    table = axes[1, 1].table(cellText=summary_data[1:], colLabels=summary_data[0], 
                           cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    axes[1, 1].set_title('Correlation Summary', pad=20)
    
    plt.suptitle('Compass Alignment Correlation Analysis', fontsize=16, y=0.98)
    plt.tight_layout()
    return fig


def calculate_compass_metrics_for_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate compass alignment metrics for all circuits in a dataset.
    
    Args:
        df: DataFrame with circuit_data and quality_label columns
        
    Returns:
        DataFrame with added compass alignment columns
    """
    from .calculate_scores import (calculate_pca_compass_alignment, 
                                 calculate_probe_compass_alignment,
                                 analyze_compass_performance)
    
    # Filter for valid data
    valid_circuits = df[df['circuit_data'].notna() & df['quality_label'].notna()].copy()
    
    if len(valid_circuits) < 3:
        print("Not enough valid circuits for compass analysis")
        return df
    
    circuits_data = valid_circuits['circuit_data'].tolist()
    quality_labels = valid_circuits['quality_label'].tolist()
    
    # Calculate PCA compass alignment
    print("Calculating PCA compass alignment...")
    pca_result = calculate_pca_compass_alignment(circuits_data, quality_labels)
    
    # Calculate probe compass alignment
    print("Calculating probe compass alignment...")
    probe_result = calculate_probe_compass_alignment(circuits_data, quality_labels)
    
    # Initialize alignment columns
    df['pca_alignment'] = np.nan
    df['probe_alignment'] = np.nan
    
    # Add PCA alignment scores
    if pca_result is not None:
        df.loc[valid_circuits.index, 'pca_alignment'] = pca_result['alignment_scores']
        print(f"PCA compass: {pca_result['n_high_quality_circuits']} high-quality circuits used")
        print(f"Explained variance ratio: {pca_result['explained_variance_ratio']:.3f}")
        
        # Analyze performance
        pca_performance = analyze_compass_performance(pca_result['alignment_scores'], quality_labels)
        print(f"PCA-Quality correlation: r={pca_performance.get('pearson_r', 'N/A'):.3f}")
    
    # Add probe alignment scores
    if probe_result is not None:
        df.loc[valid_circuits.index, 'probe_alignment'] = probe_result['alignment_scores']
        print(f"Probe compass: {probe_result['n_training_samples']} training samples")
        print(f"Training accuracy: {probe_result['training_accuracy']:.3f}")
        print(f"Good/Bad samples: {probe_result['n_good_samples']}/{probe_result['n_bad_samples']}")
        
        # Analyze performance
        probe_performance = analyze_compass_performance(probe_result['alignment_scores'], quality_labels)
        print(f"Probe-Quality correlation: r={probe_performance.get('pearson_r', 'N/A'):.3f}")
    
    return df


def calculate_advanced_metrics_for_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate advanced geometric metrics (CKA, Procrustes, aggregates) for all circuits.
    
    Args:
        df: DataFrame with circuit_data and quality_label columns
        
    Returns:
        DataFrame with added advanced metric columns
    """
    from .calculate_scores import (calculate_circuit_aggregates,
                                 calculate_circuit_cka_similarity,
                                 calculate_circuit_procrustes_disparity)
    
    # Filter for valid data
    valid_circuits = df[df['circuit_data'].notna() & df['quality_label'].notna()].copy()
    
    if len(valid_circuits) < 3:
        print("Not enough valid circuits for advanced analysis")
        return df
    
    circuits_data = valid_circuits['circuit_data'].tolist()
    quality_labels = valid_circuits['quality_label'].tolist()
    
    # Initialize new columns
    df['circuit_mean'] = np.nan
    df['circuit_max'] = np.nan
    df['circuit_min'] = np.nan
    df['circuit_range'] = np.nan
    df['circuit_sum'] = np.nan
    df['cka_similarity'] = np.nan
    df['procrustes_disparity'] = np.nan
    
    # Calculate circuit aggregates for each individual circuit
    print("Calculating circuit aggregate statistics...")
    for idx in valid_circuits.index:
        circuit_data = valid_circuits.loc[idx, 'circuit_data']
        aggregates = calculate_circuit_aggregates(circuit_data)
        
        for key, value in aggregates.items():
            df.loc[idx, key] = value
    
    # Calculate CKA similarities
    print("Calculating CKA similarities...")
    cka_scores = calculate_circuit_cka_similarity(circuits_data, quality_labels)
    if cka_scores is not None:
        df.loc[valid_circuits.index, 'cka_similarity'] = cka_scores
        print(f"CKA similarities calculated for {len(cka_scores)} circuits")
    
    # Calculate Procrustes disparities
    print("Calculating Procrustes disparities...")
    procrustes_scores = calculate_circuit_procrustes_disparity(circuits_data, quality_labels)
    if procrustes_scores is not None:
        df.loc[valid_circuits.index, 'procrustes_disparity'] = procrustes_scores
        print(f"Procrustes disparities calculated for {len(procrustes_scores)} circuits")
    
    return df


def random_forest_analysis(df: pd.DataFrame, 
                          feature_columns: List[str] = None,
                          target_column: str = 'quality_label',
                          test_size: float = 0.3) -> Dict[str, Any]:
    """Perform Random Forest analysis with permutation importance.
    
    Args:
        df: DataFrame with features and target
        feature_columns: List of feature column names to use
        target_column: Name of target column to predict
        test_size: Proportion of data to use for testing
        
    Returns:
        Dictionary with model performance and feature importance results
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.inspection import permutation_importance
        from sklearn.preprocessing import StandardScaler
        
        if feature_columns is None:
            # Use common numerical features
            feature_columns = [
                'skewness', 'kurtosis', 'avg_jaccard_similarity', 'circuit_mean', 
                'circuit_max', 'circuit_range', 'pca_alignment', 'probe_alignment',
                'cka_similarity', 'procrustes_disparity'
            ]
        
        # Filter to available features
        available_features = [col for col in feature_columns if col in df.columns]
        if len(available_features) < 2:
            return {'error': 'Not enough available features for Random Forest analysis'}
        
        # Prepare data
        ml_df = df.dropna(subset=available_features + [target_column])
        if len(ml_df) < 10:
            return {'error': 'Not enough valid samples for Random Forest analysis'}
        
        X = ml_df[available_features]
        y = ml_df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        
        # Calculate scores
        train_score = rf.score(X_train_scaled, y_train)
        test_score = rf.score(X_test_scaled, y_test)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            rf, X_test_scaled, y_test, n_repeats=10, random_state=42
        )
        
        # Organize results
        feature_importance = []
        for i, feature in enumerate(available_features):
            feature_importance.append({
                'feature': feature,
                'importance_mean': float(perm_importance.importances_mean[i]),
                'importance_std': float(perm_importance.importances_std[i])
            })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance_mean'], reverse=True)
        
        return {
            'train_r2': float(train_score),
            'test_r2': float(test_score),
            'n_features': len(available_features),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'feature_importance': feature_importance,
            'features_used': available_features
        }
        
    except Exception as e:
        return {'error': f'Random Forest analysis failed: {str(e)}'}


def plot_random_forest_importance(rf_results: Dict[str, Any],
                                 figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """Plot Random Forest feature importance results.
    
    Args:
        rf_results: Results dictionary from random_forest_analysis
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    if 'error' in rf_results:
        raise ValueError(f"Cannot plot: {rf_results['error']}")
    
    importance_data = rf_results['feature_importance']
    features = [item['feature'] for item in importance_data]
    means = [item['importance_mean'] for item in importance_data]
    stds = [item['importance_std'] for item in importance_data]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot of feature importance
    y_pos = np.arange(len(features))
    axes[0].barh(y_pos, means, xerr=stds, alpha=0.7, capsize=3)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(features)
    axes[0].set_xlabel('Permutation Importance')
    axes[0].set_title('Random Forest Feature Importance')
    axes[0].grid(True, alpha=0.3)
    
    # Model performance text
    axes[1].axis('off')
    performance_text = [
        f"Model Performance:",
        f"",
        f"Training R²: {rf_results['train_r2']:.3f}",
        f"Test R²: {rf_results['test_r2']:.3f}",
        f"",
        f"Dataset Info:",
        f"Features used: {rf_results['n_features']}",
        f"Training samples: {rf_results['n_train_samples']}",
        f"Test samples: {rf_results['n_test_samples']}",
        f"",
        f"Top 3 Features:",
    ]
    
    # Add top 3 features
    for i, item in enumerate(importance_data[:3]):
        performance_text.append(
            f"{i+1}. {item['feature']}: {item['importance_mean']:.3f}"
        )
    
    axes[1].text(0.1, 0.9, '\n'.join(performance_text), 
                transform=axes[1].transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig


def umap_clustering_analysis(df: pd.DataFrame, 
                            feature_columns: List[str] = None,
                            color_column: str = 'quality_label',
                            n_neighbors: int = 15,
                            min_dist: float = 0.1) -> Tuple[plt.Figure, np.ndarray]:
    """Perform UMAP dimensionality reduction and clustering analysis.
    
    Args:
        df: DataFrame with features
        feature_columns: List of feature column names to use
        color_column: Column to use for coloring points
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        
    Returns:
        Tuple of (matplotlib figure, UMAP embedding array)
    """
    try:
        import umap.umap_ as umap
        from sklearn.preprocessing import StandardScaler
        
        if feature_columns is None:
            # Use numerical features available
            feature_columns = [
                'skewness', 'kurtosis', 'avg_jaccard_similarity', 'circuit_mean',
                'circuit_max', 'circuit_range', 'pca_alignment', 'probe_alignment',
                'cka_similarity', 'procrustes_disparity'
            ]
        
        # Filter to available features
        available_features = [col for col in feature_columns if col in df.columns]
        if len(available_features) < 2:
            raise ValueError('Not enough available features for UMAP analysis')
        
        # Prepare data
        analysis_df = df.dropna(subset=available_features + [color_column])
        if len(analysis_df) < 10:
            raise ValueError('Not enough valid samples for UMAP analysis')
        
        X = analysis_df[available_features].values
        colors = analysis_df[color_column].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit UMAP
        reducer = umap.UMAP(
            n_neighbors=min(n_neighbors, len(X_scaled)-1),
            min_dist=min_dist,
            n_components=2,
            random_state=42
        )
        embedding = reducer.fit_transform(X_scaled)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # UMAP scatter plot
        scatter = axes[0].scatter(
            embedding[:, 0], embedding[:, 1], 
            c=colors, cmap='viridis', alpha=0.7, s=50
        )
        axes[0].set_xlabel('UMAP Dimension 1')
        axes[0].set_ylabel('UMAP Dimension 2')
        axes[0].set_title(f'UMAP Projection (Colored by {color_column})')
        plt.colorbar(scatter, ax=axes[0])
        axes[0].grid(True, alpha=0.3)
        
        # Quality distribution in UMAP space
        unique_qualities = sorted(analysis_df[color_column].unique())
        for quality in unique_qualities:
            mask = colors == quality
            if np.any(mask):
                axes[1].scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    label=f'{color_column} {quality}', alpha=0.7, s=50
                )
        
        axes[1].set_xlabel('UMAP Dimension 1')
        axes[1].set_ylabel('UMAP Dimension 2')
        axes[1].set_title(f'UMAP Projection by {color_column} Groups')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, embedding
        
    except Exception as e:
        raise ValueError(f'UMAP analysis failed: {str(e)}')


def dbscan_clustering_analysis(df: pd.DataFrame,
                              feature_columns: List[str] = None,
                              eps: float = 0.5,
                              min_samples: int = 5) -> Tuple[plt.Figure, np.ndarray]:
    """Perform DBSCAN clustering analysis on circuit features.
    
    Args:
        df: DataFrame with features
        feature_columns: List of feature column names to use
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples parameter
        
    Returns:
        Tuple of (matplotlib figure, cluster labels array)
    """
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        
        if feature_columns is None:
            feature_columns = [
                'skewness', 'kurtosis', 'avg_jaccard_similarity', 'circuit_mean',
                'circuit_max', 'circuit_range', 'pca_alignment', 'probe_alignment',
                'cka_similarity', 'procrustes_disparity'
            ]
        
        # Filter to available features
        available_features = [col for col in feature_columns if col in df.columns]
        if len(available_features) < 2:
            raise ValueError('Not enough available features for DBSCAN analysis')
        
        # Prepare data
        analysis_df = df.dropna(subset=available_features + ['quality_label'])
        if len(analysis_df) < 10:
            raise ValueError('Not enough valid samples for DBSCAN analysis')
        
        X = analysis_df[available_features].values
        quality_labels = analysis_df['quality_label'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(X_scaled)
        
        # Analyze clustering results
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Use UMAP for 2D visualization if available
        try:
            import umap.umap_ as umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(X_scaled)
            x_coord, y_coord = embedding[:, 0], embedding[:, 1]
            coord_label = 'UMAP'
        except ImportError:
            # Fallback to first two features
            x_coord, y_coord = X_scaled[:, 0], X_scaled[:, 1]
            coord_label = f'{available_features[0]} vs {available_features[1]}'
        
        # Plot 1: Clusters
        scatter1 = axes[0, 0].scatter(x_coord, y_coord, c=cluster_labels, 
                                     cmap='viridis', alpha=0.7, s=50)
        axes[0, 0].set_xlabel(f'{coord_label} Dimension 1')
        axes[0, 0].set_ylabel(f'{coord_label} Dimension 2')
        axes[0, 0].set_title(f'DBSCAN Clusters (n={n_clusters}, noise={n_noise})')
        plt.colorbar(scatter1, ax=axes[0, 0])
        
        # Plot 2: Quality labels
        scatter2 = axes[0, 1].scatter(x_coord, y_coord, c=quality_labels, 
                                     cmap='plasma', alpha=0.7, s=50)
        axes[0, 1].set_xlabel(f'{coord_label} Dimension 1')
        axes[0, 1].set_ylabel(f'{coord_label} Dimension 2')
        axes[0, 1].set_title('True Quality Labels')
        plt.colorbar(scatter2, ax=axes[0, 1])
        
        # Plot 3: Cluster vs Quality analysis
        if n_clusters > 0:
            cluster_quality_analysis = []
            for cluster_id in set(cluster_labels):
                if cluster_id != -1:  # Ignore noise
                    cluster_mask = cluster_labels == cluster_id
                    cluster_qualities = quality_labels[cluster_mask]
                    cluster_quality_analysis.append({
                        'cluster': cluster_id,
                        'size': np.sum(cluster_mask),
                        'mean_quality': np.mean(cluster_qualities),
                        'std_quality': np.std(cluster_qualities)
                    })
            
            if cluster_quality_analysis:
                cluster_ids = [item['cluster'] for item in cluster_quality_analysis]
                mean_qualities = [item['mean_quality'] for item in cluster_quality_analysis]
                std_qualities = [item['std_quality'] for item in cluster_quality_analysis]
                
                axes[1, 0].bar(cluster_ids, mean_qualities, yerr=std_qualities, 
                              alpha=0.7, capsize=5)
                axes[1, 0].set_xlabel('Cluster ID')
                axes[1, 0].set_ylabel('Mean Quality Label')
                axes[1, 0].set_title('Average Quality by Cluster')
                axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        axes[1, 1].axis('off')
        summary_text = [
            f"DBSCAN Clustering Results:",
            f"",
            f"Parameters:",
            f"  eps = {eps}",
            f"  min_samples = {min_samples}",
            f"",
            f"Results:",
            f"  Number of clusters: {n_clusters}",
            f"  Number of noise points: {n_noise}",
            f"  Total samples: {len(cluster_labels)}",
            f"",
            f"Features used ({len(available_features)}):",
        ]
        
        for feature in available_features[:5]:  # Show first 5 features
            summary_text.append(f"  • {feature}")
        
        if len(available_features) > 5:
            summary_text.append(f"  ... and {len(available_features)-5} more")
        
        axes[1, 1].text(0.1, 0.9, '\n'.join(summary_text),
                       transform=axes[1, 1].transAxes, fontsize=10,
                       verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        return fig, cluster_labels
        
    except Exception as e:
        raise ValueError(f'DBSCAN analysis failed: {str(e)}')