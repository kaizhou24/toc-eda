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