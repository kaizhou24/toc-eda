# Tree of Circuits Analysis

A comprehensive toolkit for analyzing circuit data with quality metrics and visualization capabilities.

## Project Structure

```
tree-of-circuits/
├── src/                    # Core analysis modules
│   ├── __init__.py        # Package initialization
│   ├── calculate_scores.py # Metric calculations (skewness, Jaccard similarity, etc.)
│   ├── data_loader.py     # Data loading and preprocessing
│   └── visualize_scores.py # Visualization functions with log transformations
├── tests/                 # Test scripts
│   ├── test_visualizations.py
│   └── test_metric_relationships.py
├── outputs/              # Generated plots and analysis results
├── data/                # Input data files
└── pyproject.toml       # Project configuration
```

## Key Features

### Metric Calculations (`src/calculate_scores.py`)
- **Basic Statistics**: mean, std, skewness, kurtosis, percentiles
- **Jaccard Similarity**: Circuit similarity based on edge presence above threshold
- **Average Jaccard Similarity**: Mean similarity excluding self-comparisons
- **Distribution Analysis**: Histogram-based score distributions

### Visualization (`src/visualize_scores.py`)
- **Symmetric Log Transformation**: `f(x) = sign(x) * log(1 + |x|)` for extreme value compression
- **Quality Label Clustering**: Color-coded scatter plots and box plots by quality
- **Metric Relationships**: Pairwise analysis of metrics with quality separation
- **Distribution Comparisons**: Raw vs log-transformed data visualization

### Data Processing (`src/data_loader.py`)
- **Unified Dataset Creation**: Matches circuit files with thought labels
- **Robust File Handling**: Graceful handling of missing or corrupted data
- **Flexible Data Loading**: Support for multiple dataset formats

## Usage Examples

### Basic Metric Analysis
```python
from src import calculate_metrics_for_dataset, load_and_build_dataframe

# Load your dataset
df = load_and_build_dataframe("test2", base_path="data")

# Calculate all metrics including Jaccard similarity
df_metrics = calculate_metrics_for_dataset(df)
```

### Quality Clustering Visualization
```python
from src import plot_metric_relationship_with_labels

# Create skewness vs Jaccard similarity plot with quality labels
fig = plot_metric_relationship_with_labels(
    df_metrics, 
    'avg_jaccard_similarity', 'skewness',
    apply_log_transform=True,
    color_by_quality=True
)
```

### Comprehensive Analysis
```python
from src import plot_quality_clustering_analysis, plot_pairwise_metric_relationships

# Quality clustering analysis across multiple metrics
fig1 = plot_quality_clustering_analysis(df_metrics)

# Pairwise metric relationships matrix
fig2 = plot_pairwise_metric_relationships(df_metrics)
```

## Running Tests

```bash
# Test basic visualizations
uv run --no-project python tests/basic_visualization.py

# Test metric relationships with quality labels
uv run --no-project python tests/metric_rel_visualization.py
```

All generated plots are saved to the `outputs/` directory.

## Key Insights from Analysis

The system identifies quality patterns through:

1. **Average Jaccard Similarity**: Higher quality circuits show greater similarity to each other
2. **Standard Deviation**: Lower quality circuits exhibit higher variance
3. **Kurtosis**: Quality correlates with distribution tail behavior
4. **Log Transformation**: Essential for visualizing extreme values in circuit scores

Quality separation analysis shows Jaccard similarity as the strongest discriminator between quality levels.