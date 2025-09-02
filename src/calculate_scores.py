"""
calculate_scores.py

Functions to calculate features from circuit data.
Implementations for skewness, similarity, kurtosis, and other statistical measures.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity


def extract_edge_scores(circuit_data: Dict[str, Any]) -> List[float]:
    """Extract edge scores from circuit data.
    
    Args:
        circuit_data: Circuit dictionary with 'edges' key containing score data
        
    Returns:
        List of edge scores as floats
        
    Raises:
        ValueError: If circuit_data is None or doesn't contain 'edges'
    """
    if circuit_data is None:
        raise ValueError("Circuit data is None")
    
    if 'edges' not in circuit_data:
        raise ValueError("Circuit data missing 'edges' key")
    
    edges = circuit_data['edges']
    if not edges:
        return []
    
    scores = []
    for edge_data in edges.values():
        if isinstance(edge_data, dict) and 'score' in edge_data:
            score = edge_data['score']
            if isinstance(score, (int, float)) and np.isfinite(score):
                scores.append(float(score))
    
    return scores


def calculate_skewness(circuit_data: Dict[str, Any]) -> Optional[float]:
    """Calculate skewness from circuit edge scores.
    
    Args:
        circuit_data: Circuit dictionary with edge score data
        
    Returns:
        Skewness value as float, or None if insufficient data
    """
    try:
        scores = extract_edge_scores(circuit_data)
        if len(scores) < 3:
            return None
        return float(stats.skew(scores))
    except (ValueError, TypeError):
        return None


def calculate_kurtosis(circuit_data: Dict[str, Any]) -> Optional[float]:
    """Calculate kurtosis from circuit edge scores.
    
    Args:
        circuit_data: Circuit dictionary with edge score data
        
    Returns:
        Kurtosis value as float, or None if insufficient data
    """
    try:
        scores = extract_edge_scores(circuit_data)
        if len(scores) < 4:
            return None
        return float(stats.kurtosis(scores))
    except (ValueError, TypeError):
        return None


def calculate_basic_stats(circuit_data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Calculate basic statistical measures from circuit edge scores.
    
    Args:
        circuit_data: Circuit dictionary with edge score data
        
    Returns:
        Dictionary with keys: mean, std, min, max, median, skewness, kurtosis
    """
    try:
        scores = extract_edge_scores(circuit_data)
        if not scores:
            return {key: None for key in ['mean', 'std', 'min', 'max', 'median', 'skewness', 'kurtosis']}
        
        scores_array = np.array(scores)
        result = {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'median': float(np.median(scores_array)),
        }
        
        if len(scores) >= 3:
            result['skewness'] = float(stats.skew(scores_array))
        else:
            result['skewness'] = None
            
        if len(scores) >= 4:
            result['kurtosis'] = float(stats.kurtosis(scores_array))
        else:
            result['kurtosis'] = None
            
        return result
    except (ValueError, TypeError):
        return {key: None for key in ['mean', 'std', 'min', 'max', 'median', 'skewness', 'kurtosis']}


def calculate_similarity(circuit_data1: Dict[str, Any], circuit_data2: Dict[str, Any], 
                        method: str = 'cosine') -> Optional[float]:
    """Calculate similarity between two circuits based on their edge scores.
    
    Args:
        circuit_data1: First circuit dictionary
        circuit_data2: Second circuit dictionary  
        method: Similarity method ('cosine', 'pearson', 'spearman')
        
    Returns:
        Similarity score as float, or None if calculation fails
    """
    try:
        scores1 = extract_edge_scores(circuit_data1)
        scores2 = extract_edge_scores(circuit_data2)
        
        if not scores1 or not scores2:
            return None
            
        if method == 'cosine':
            return float(cosine_similarity([scores1], [scores2])[0, 0])
        elif method == 'pearson':
            corr, _ = stats.pearsonr(scores1, scores2)
            return float(corr) if np.isfinite(corr) else None
        elif method == 'spearman':
            corr, _ = stats.spearmanr(scores1, scores2)
            return float(corr) if np.isfinite(corr) else None
        else:
            raise ValueError(f"Unknown similarity method: {method}")
            
    except (ValueError, TypeError):
        return None


def calculate_score_distribution(circuit_data: Dict[str, Any], 
                               bins: int = 10) -> Optional[Dict[str, Any]]:
    """Calculate distribution of edge scores.
    
    Args:
        circuit_data: Circuit dictionary with edge score data
        bins: Number of bins for histogram
        
    Returns:
        Dictionary with histogram data or None if calculation fails
    """
    try:
        scores = extract_edge_scores(circuit_data)
        if not scores:
            return None
            
        hist, bin_edges = np.histogram(scores, bins=bins)
        return {
            'histogram': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'bin_centers': ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
        }
    except (ValueError, TypeError):
        return None


def calculate_jaccard_similarity(circuit_data1: Dict[str, Any], circuit_data2: Dict[str, Any],
                                threshold: Optional[float] = None) -> Optional[float]:
    """Calculate Jaccard similarity between two circuits based on edge presence above threshold.
    
    Args:
        circuit_data1: First circuit dictionary
        circuit_data2: Second circuit dictionary
        threshold: Score threshold for considering edge as present. If None, uses median of combined scores
        
    Returns:
        Jaccard similarity coefficient (intersection/union), or None if calculation fails
    """
    try:
        scores1 = extract_edge_scores(circuit_data1)
        scores2 = extract_edge_scores(circuit_data2)
        
        if not scores1 or not scores2:
            return None
            
        # Use median of combined scores as threshold if not provided
        if threshold is None:
            all_scores = scores1 + scores2
            threshold = float(np.median(all_scores))
        
        # Convert to binary sets based on threshold
        set1 = set(i for i, score in enumerate(scores1) if score > threshold)
        set2 = set(i for i, score in enumerate(scores2) if score > threshold)
        
        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return float(intersection / union)
        
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def calculate_average_jaccard_similarity(circuit_data: Dict[str, Any], 
                                       other_circuits: List[Dict[str, Any]],
                                       exclude_self: bool = True) -> Optional[float]:
    """Calculate average Jaccard similarity between a circuit and a list of other circuits.
    
    Args:
        circuit_data: Target circuit dictionary
        other_circuits: List of other circuit dictionaries to compare against
        exclude_self: Whether to exclude identical circuits from calculation
        
    Returns:
        Average Jaccard similarity, or None if calculation fails
    """
    try:
        if not other_circuits:
            return None
            
        similarities = []
        for other_circuit in other_circuits:
            if exclude_self and circuit_data is other_circuit:
                continue
                
            similarity = calculate_jaccard_similarity(circuit_data, other_circuit)
            if similarity is not None:
                similarities.append(similarity)
        
        if not similarities:
            return None
            
        return float(np.mean(similarities))
        
    except (ValueError, TypeError):
        return None


def calculate_percentiles(circuit_data: Dict[str, Any], 
                         percentiles: List[float] = [25, 50, 75, 90, 95, 99]) -> Dict[str, Optional[float]]:
    """Calculate percentiles of edge scores.
    
    Args:
        circuit_data: Circuit dictionary with edge score data
        percentiles: List of percentiles to calculate
        
    Returns:
        Dictionary mapping percentile to value
    """
    try:
        scores = extract_edge_scores(circuit_data)
        if not scores:
            return {f'p{p}': None for p in percentiles}
            
        result = {}
        for p in percentiles:
            result[f'p{p}'] = float(np.percentile(scores, p))
        return result
    except (ValueError, TypeError):
        return {f'p{p}': None for p in percentiles}


def calculate_pca_compass_alignment(circuits_data: List[Dict[str, Any]], 
                                  quality_labels: List[int],
                                  high_quality_threshold: int = 7) -> Optional[Dict[str, Any]]:
    """Calculate PCA-based compass alignment for circuit quality analysis.
    
    Uses Principal Component Analysis to find the primary direction of variance 
    among high-quality circuits, then projects all circuits onto this direction.
    
    Args:
        circuits_data: List of circuit dictionaries with edge score data
        quality_labels: List of quality labels corresponding to circuits
        high_quality_threshold: Minimum quality label to consider "high quality"
        
    Returns:
        Dictionary with alignment scores, PCA components, and analysis metadata
    """
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
    except ImportError:
        return None
    
    try:
        if len(circuits_data) != len(quality_labels):
            raise ValueError("circuits_data and quality_labels must have same length")
        
        # Extract edge scores for all circuits
        all_circuit_vectors = []
        for circuit_data in circuits_data:
            scores = extract_edge_scores(circuit_data)
            if not scores:
                return None
            all_circuit_vectors.append(scores)
        
        # Ensure all vectors have the same length by padding with zeros
        max_length = max(len(vec) for vec in all_circuit_vectors)
        padded_vectors = []
        for vec in all_circuit_vectors:
            padded = vec + [0.0] * (max_length - len(vec))
            padded_vectors.append(padded)
        
        all_activations = np.array(padded_vectors)
        
        # Filter for high-quality circuits
        high_quality_mask = np.array(quality_labels) >= high_quality_threshold
        if not np.any(high_quality_mask):
            return None
        
        high_quality_activations = all_activations[high_quality_mask]
        
        # Scale the high-quality data
        scaler = StandardScaler()
        scaled_high_quality = scaler.fit_transform(high_quality_activations)
        
        # Fit PCA on high-quality circuits
        pca = PCA(n_components=min(10, len(scaled_high_quality)))
        pca.fit(scaled_high_quality)
        
        # Get the first principal component (compass direction)
        pca_direction = pca.components_[0]
        
        # Scale all activations using the same scaler
        all_activations_scaled = scaler.transform(all_activations)
        
        # Calculate alignment scores for all circuits
        alignment_scores = all_activations_scaled @ pca_direction
        
        return {
            'alignment_scores': alignment_scores.tolist(),
            'pca_direction': pca_direction.tolist(),
            'explained_variance_ratio': float(pca.explained_variance_ratio_[0]),
            'n_high_quality_circuits': int(np.sum(high_quality_mask)),
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist()
        }
        
    except Exception:
        return None


def calculate_probe_compass_alignment(circuits_data: List[Dict[str, Any]], 
                                    quality_labels: List[int],
                                    good_threshold: int = 7,
                                    bad_threshold: int = 3) -> Optional[Dict[str, Any]]:
    """Calculate supervised probe-based compass alignment for circuit quality analysis.
    
    Trains a linear classifier to distinguish between high and low quality circuits,
    then uses the learned weights as a compass direction.
    
    Args:
        circuits_data: List of circuit dictionaries with edge score data
        quality_labels: List of quality labels corresponding to circuits
        good_threshold: Minimum quality label to consider "good"
        bad_threshold: Maximum quality label to consider "bad"
        
    Returns:
        Dictionary with alignment scores, probe weights, and training metadata
    """
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    
    try:
        if len(circuits_data) != len(quality_labels):
            raise ValueError("circuits_data and quality_labels must have same length")
        
        # Extract edge scores for all circuits
        all_circuit_vectors = []
        for circuit_data in circuits_data:
            scores = extract_edge_scores(circuit_data)
            if not scores:
                return None
            all_circuit_vectors.append(scores)
        
        # Ensure all vectors have the same length by padding with zeros
        max_length = max(len(vec) for vec in all_circuit_vectors)
        padded_vectors = []
        for vec in all_circuit_vectors:
            padded = vec + [0.0] * (max_length - len(vec))
            padded_vectors.append(padded)
        
        all_activations = np.array(padded_vectors)
        quality_array = np.array(quality_labels)
        
        # Create binary labels for probe training
        binary_labels = []
        training_indices = []
        
        for i, quality in enumerate(quality_labels):
            if quality <= bad_threshold:
                binary_labels.append(0)
                training_indices.append(i)
            elif quality >= good_threshold:
                binary_labels.append(1)
                training_indices.append(i)
        
        if len(training_indices) < 2:
            return None
        
        # Prepare training data
        training_activations = all_activations[training_indices]
        training_labels = np.array(binary_labels)
        
        # Scale the training data
        scaler = StandardScaler()
        training_scaled = scaler.fit_transform(training_activations)
        
        # Train the probe
        probe = LogisticRegression(class_weight='balanced', random_state=42)
        probe.fit(training_scaled, training_labels)
        
        # Get probe direction (weights)
        probe_direction = probe.coef_[0]
        
        # Scale all activations and calculate alignment
        all_activations_scaled = scaler.transform(all_activations)
        alignment_scores = all_activations_scaled @ probe_direction
        
        # Calculate training accuracy
        training_accuracy = probe.score(training_scaled, training_labels)
        
        return {
            'alignment_scores': alignment_scores.tolist(),
            'probe_direction': probe_direction.tolist(),
            'training_accuracy': float(training_accuracy),
            'n_training_samples': len(training_indices),
            'n_good_samples': int(np.sum(training_labels == 1)),
            'n_bad_samples': int(np.sum(training_labels == 0)),
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist()
        }
        
    except Exception:
        return None


def analyze_compass_performance(alignment_scores: List[float], 
                              quality_labels: List[int]) -> Dict[str, float]:
    """Analyze how well compass alignment correlates with quality labels.
    
    Args:
        alignment_scores: List of alignment scores from compass analysis
        quality_labels: List of corresponding quality labels
        
    Returns:
        Dictionary with correlation statistics
    """
    try:
        from scipy.stats import pearsonr, spearmanr
        
        if len(alignment_scores) != len(quality_labels):
            raise ValueError("alignment_scores and quality_labels must have same length")
        
        # Remove any NaN values
        valid_pairs = [(score, label) for score, label in zip(alignment_scores, quality_labels) 
                      if np.isfinite(score)]
        
        if len(valid_pairs) < 3:
            return {'pearson_r': None, 'pearson_p': None, 'spearman_r': None, 'spearman_p': None}
        
        scores, labels = zip(*valid_pairs)
        
        # Calculate correlations
        pearson_r, pearson_p = pearsonr(scores, labels)
        spearman_r, spearman_p = spearmanr(scores, labels)
        
        return {
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'n_valid_samples': len(valid_pairs)
        }
        
    except Exception:
        return {'pearson_r': None, 'pearson_p': None, 'spearman_r': None, 'spearman_p': None}


# Example usage and testing functions
if __name__ == '__main__':
    from data_loader import load_and_build_dataframe
    
    print("Loading test2 dataset for demonstration...")
    df = load_and_build_dataframe('test2')
    
    # Find records with circuit data
    circuits_df = df[df['circuit_data'].notna()]
    if len(circuits_df) >= 2:
        print(f"\n--- Testing calculate_scores.py functions ---")
        print(f"Found {len(circuits_df)} records with circuit data")
        
        # Test basic stats on first circuit
        first_circuit = circuits_df.iloc[0]['circuit_data']
        stats_result = calculate_basic_stats(first_circuit)
        print(f"\nBasic stats for first circuit:")
        for key, value in stats_result.items():
            if value is not None:
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: None")
        
        # Test individual functions
        skew_val = calculate_skewness(first_circuit)
        kurt_val = calculate_kurtosis(first_circuit)
        print(f"\nIndividual calculations:")
        print(f"  Skewness: {skew_val:.4f}" if skew_val else "  Skewness: None")
        print(f"  Kurtosis: {kurt_val:.4f}" if kurt_val else "  Kurtosis: None")
        
        # Test percentiles
        percentiles = calculate_percentiles(first_circuit)
        print(f"\nPercentiles:")
        for key, value in percentiles.items():
            if value is not None:
                print(f"  {key}: {value:.2f}")
        
        # Test similarity if we have multiple circuits
        if len(circuits_df) >= 2:
            second_circuit = circuits_df.iloc[1]['circuit_data']
            similarity = calculate_similarity(first_circuit, second_circuit, method='cosine')
            print(f"\nCosine similarity between first two circuits: {similarity:.4f}" if similarity else "Similarity: None")
        
        # Test distribution
        distribution = calculate_score_distribution(first_circuit, bins=5)
        if distribution:
            print(f"\nScore distribution (5 bins):")
            for i, (count, center) in enumerate(zip(distribution['histogram'], distribution['bin_centers'])):
                print(f"  Bin {i+1} (center {center:.2f}): {count} scores")
    else:
        print("⚠️ No circuit data found for testing")