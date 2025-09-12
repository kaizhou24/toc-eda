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


def calculate_mlp_compass_alignment(circuits_data: List[Dict[str, Any]], 
                                  quality_labels: List[int],
                                  good_threshold: int = 7,
                                  bad_threshold: int = 3,
                                  hidden_layer_sizes: Tuple[int, ...] = (64, 32)) -> Optional[Dict[str, Any]]:
    """Calculate MLP-based compass alignment for circuit quality analysis.
    
    Trains a multi-layer perceptron to distinguish between high and low quality circuits,
    then uses the decision boundary gradients as a compass direction.
    
    Args:
        circuits_data: List of circuit dictionaries with edge score data
        quality_labels: List of quality labels corresponding to circuits
        good_threshold: Minimum quality label to consider "good"
        bad_threshold: Maximum quality label to consider "bad"
        hidden_layer_sizes: Hidden layer architecture for MLP
        
    Returns:
        Dictionary with alignment scores, training metadata
    """
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier
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
        
        if len(training_indices) < 4:  # Need more samples for MLP
            return None
        
        # Prepare training data
        training_activations = all_activations[training_indices]
        training_labels = np.array(binary_labels)
        
        # Scale the training data
        scaler = StandardScaler()
        training_scaled = scaler.fit_transform(training_activations)
        
        # Train the MLP probe
        mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                           activation='relu',
                           solver='adam',
                           alpha=0.001,
                           learning_rate='constant',
                           max_iter=1000,
                           random_state=42)
        mlp.fit(training_scaled, training_labels)
        
        # Scale all activations and get prediction probabilities
        all_activations_scaled = scaler.transform(all_activations)
        
        # For alignment, use the probability of being "good" class
        # This gives a continuous measure of how "good" each circuit is
        probabilities = mlp.predict_proba(all_activations_scaled)
        # Convert probabilities to alignment scores (difference from neutral)
        alignment_scores = probabilities[:, 1] - 0.5
        
        # Calculate training accuracy
        training_accuracy = mlp.score(training_scaled, training_labels)
        
        return {
            'alignment_scores': alignment_scores.tolist(),
            'training_accuracy': float(training_accuracy),
            'n_training_samples': len(training_indices),
            'n_good_samples': int(np.sum(training_labels == 1)),
            'n_bad_samples': int(np.sum(training_labels == 0)),
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'hidden_layer_sizes': hidden_layer_sizes,
            'n_features': int(all_activations.shape[1])
        }
        
    except Exception:
        return None


def calculate_svm_compass_alignment(circuits_data: List[Dict[str, Any]], 
                                  quality_labels: List[int],
                                  good_threshold: int = 7,
                                  bad_threshold: int = 3,
                                  kernel: str = 'rbf',
                                  gamma: str = 'scale') -> Optional[Dict[str, Any]]:
    """Calculate kernel SVM-based compass alignment for circuit quality analysis.
    
    Trains a support vector machine with non-linear kernel to distinguish between 
    high and low quality circuits, then uses decision function scores for alignment.
    
    Args:
        circuits_data: List of circuit dictionaries with edge score data
        quality_labels: List of quality labels corresponding to circuits
        good_threshold: Minimum quality label to consider "good"
        bad_threshold: Maximum quality label to consider "bad"
        kernel: Kernel type ('rbf', 'poly', 'sigmoid')
        gamma: Kernel coefficient
        
    Returns:
        Dictionary with alignment scores, training metadata
    """
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
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
        
        if len(training_indices) < 4:  # Need more samples for SVM
            return None
        
        # Prepare training data
        training_activations = all_activations[training_indices]
        training_labels = np.array(binary_labels)
        
        # Scale the training data
        scaler = StandardScaler()
        training_scaled = scaler.fit_transform(training_activations)
        
        # Train the SVM probe
        svm = SVC(kernel=kernel, 
                  gamma=gamma,
                  class_weight='balanced',
                  probability=False,  # We'll use decision_function
                  random_state=42)
        svm.fit(training_scaled, training_labels)
        
        # Scale all activations and get decision function scores
        all_activations_scaled = scaler.transform(all_activations)
        
        # Use decision function for alignment scores
        alignment_scores = svm.decision_function(all_activations_scaled)
        
        # Calculate training accuracy
        training_accuracy = svm.score(training_scaled, training_labels)
        
        return {
            'alignment_scores': alignment_scores.tolist(),
            'training_accuracy': float(training_accuracy),
            'n_training_samples': len(training_indices),
            'n_good_samples': int(np.sum(training_labels == 1)),
            'n_bad_samples': int(np.sum(training_labels == 0)),
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'kernel': kernel,
            'gamma': gamma,
            'n_support_vectors': int(svm.n_support_.sum()),
            'n_features': int(all_activations.shape[1])
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


def calculate_circuit_aggregates(circuit_data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Calculate explicit mean/max aggregated moments per circuit.
    
    Args:
        circuit_data: Circuit dictionary with edge score data
        
    Returns:
        Dictionary with mean, max, min, range, and other aggregate statistics
    """
    try:
        scores = extract_edge_scores(circuit_data)
        if not scores:
            return {'circuit_mean': None, 'circuit_max': None, 'circuit_min': None, 
                   'circuit_range': None, 'circuit_sum': None}
        
        scores_array = np.array(scores)
        return {
            'circuit_mean': float(np.mean(scores_array)),
            'circuit_max': float(np.max(scores_array)),
            'circuit_min': float(np.min(scores_array)),
            'circuit_range': float(np.max(scores_array) - np.min(scores_array)),
            'circuit_sum': float(np.sum(scores_array))
        }
        
    except (ValueError, TypeError):
        return {'circuit_mean': None, 'circuit_max': None, 'circuit_min': None,
               'circuit_range': None, 'circuit_sum': None}


def centered_kernel_alignment(X: np.ndarray, Y: np.ndarray) -> Optional[float]:
    """Compute Centered Kernel Alignment (CKA) between two matrices.
    
    CKA measures representational similarity between high-dimensional vectors,
    answering: "Do these two sets encode similar relationships between data points?"
    
    Args:
        X: First matrix (n_samples, n_features)
        Y: Second matrix (n_samples, n_features) 
        
    Returns:
        CKA score (0-1, higher means more similar representations)
    """
    try:
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have same number of samples")
        
        # Compute Gram matrices
        K = X @ X.T
        L = Y @ Y.T
        
        # Center the Gram matrices
        n = K.shape[0]
        unit = np.ones((n, n))
        I = np.eye(n)
        H = I - unit / n
        
        Kc = H @ K @ H
        Lc = H @ L @ H
        
        # Compute CKA score
        numerator = np.sum(Kc * Lc)
        denominator = np.sqrt(np.sum(Kc * Kc)) * np.sqrt(np.sum(Lc * Lc))
        
        if denominator == 0:
            return None
        
        cka_score = numerator / denominator
        return float(cka_score)
        
    except Exception:
        return None


def procrustes_disparity(matrix1: np.ndarray, matrix2: np.ndarray) -> Optional[float]:
    """Calculate Procrustes disparity between two matrices.
    
    Procrustes analysis aligns two shapes by rotation, scaling, and translation,
    then measures remaining differences. Useful for comparing geometric structures.
    
    Args:
        matrix1: First matrix (n_points, n_dimensions)
        matrix2: Second matrix (n_points, n_dimensions)
        
    Returns:
        Procrustes disparity (lower means more similar shapes)
    """
    try:
        from scipy.spatial import procrustes
        _, _, disparity = procrustes(matrix1, matrix2)
        return float(disparity)
        
    except Exception:
        return None


def calculate_circuit_cka_similarity(circuits_data: List[Dict[str, Any]], 
                                   quality_labels: List[int],
                                   reference_quality_threshold: int = 7) -> Optional[List[float]]:
    """Calculate CKA similarity of each circuit to high-quality reference circuits.
    
    Args:
        circuits_data: List of circuit dictionaries
        quality_labels: List of quality labels for circuits
        reference_quality_threshold: Minimum quality to include in reference
        
    Returns:
        List of CKA similarity scores for each circuit
    """
    try:
        if len(circuits_data) != len(quality_labels):
            raise ValueError("circuits_data and quality_labels must have same length")
        
        # Extract and pad all circuit vectors
        all_vectors = []
        for circuit_data in circuits_data:
            scores = extract_edge_scores(circuit_data)
            if not scores:
                return None
            all_vectors.append(scores)
        
        # Find maximum length and pad vectors
        max_length = max(len(vec) for vec in all_vectors)
        padded_vectors = []
        for vec in all_vectors:
            padded = vec + [0.0] * (max_length - len(vec))
            padded_vectors.append(padded)
        
        all_circuits_matrix = np.array(padded_vectors)
        quality_array = np.array(quality_labels)
        
        # Create reference from high-quality circuits
        high_quality_mask = quality_array >= reference_quality_threshold
        if not np.any(high_quality_mask):
            return None
        
        reference_circuits = all_circuits_matrix[high_quality_mask]
        
        # Calculate CKA similarity for each circuit
        cka_similarities = []
        for i, circuit_vector in enumerate(all_circuits_matrix):
            # Reshape to 2D matrices for CKA
            circuit_matrix = circuit_vector.reshape(1, -1)
            reference_matrix = reference_circuits.mean(axis=0).reshape(1, -1)
            
            # For CKA to work properly, we need multiple samples
            # Create synthetic samples by adding small noise
            circuit_samples = np.vstack([
                circuit_matrix + np.random.normal(0, 0.001, circuit_matrix.shape) 
                for _ in range(10)
            ])
            reference_samples = np.vstack([
                reference_matrix + np.random.normal(0, 0.001, reference_matrix.shape) 
                for _ in range(10)
            ])
            
            cka_score = centered_kernel_alignment(circuit_samples, reference_samples)
            cka_similarities.append(cka_score if cka_score is not None else 0.0)
        
        return cka_similarities
        
    except Exception:
        return None


def calculate_circuit_procrustes_disparity(circuits_data: List[Dict[str, Any]], 
                                         quality_labels: List[int],
                                         reference_quality_threshold: int = 7) -> Optional[List[float]]:
    """Calculate Procrustes disparity of each circuit to high-quality reference shape.
    
    Args:
        circuits_data: List of circuit dictionaries
        quality_labels: List of quality labels for circuits  
        reference_quality_threshold: Minimum quality to include in reference
        
    Returns:
        List of Procrustes disparity scores for each circuit
    """
    try:
        if len(circuits_data) != len(quality_labels):
            raise ValueError("circuits_data and quality_labels must have same length")
        
        # Extract and pad all circuit vectors
        all_vectors = []
        for circuit_data in circuits_data:
            scores = extract_edge_scores(circuit_data)
            if not scores:
                return None
            all_vectors.append(scores)
        
        # Find maximum length and pad vectors
        max_length = max(len(vec) for vec in all_vectors)
        if max_length < 2:
            return None  # Need at least 2 dimensions for Procrustes
        
        padded_vectors = []
        for vec in all_vectors:
            padded = vec + [0.0] * (max_length - len(vec))
            padded_vectors.append(padded)
        
        all_circuits_matrix = np.array(padded_vectors)
        quality_array = np.array(quality_labels)
        
        # Create reference shape from high-quality circuits
        high_quality_mask = quality_array >= reference_quality_threshold
        if not np.any(high_quality_mask):
            return None
        
        reference_circuits = all_circuits_matrix[high_quality_mask]
        reference_shape = reference_circuits.mean(axis=0)
        
        # Reshape for Procrustes analysis (needs 2D: points x dimensions)
        # We'll treat each score as a point in 2D space using its index and value
        n_points = min(max_length, 100)  # Limit points for computational efficiency
        reference_points = np.column_stack([
            np.arange(n_points), 
            reference_shape[:n_points]
        ])
        
        # Calculate Procrustes disparity for each circuit
        disparities = []
        for circuit_vector in all_circuits_matrix:
            circuit_points = np.column_stack([
                np.arange(n_points),
                circuit_vector[:n_points]
            ])
            
            disparity = procrustes_disparity(reference_points, circuit_points)
            disparities.append(disparity if disparity is not None else float('inf'))
        
        return disparities
        
    except Exception:
        return None


def statistical_significance_tests(feature_values: List[float], 
                                 quality_labels: List[int],
                                 good_threshold: int = 7,
                                 bad_threshold: int = 3) -> Dict[str, Any]:
    """Perform statistical significance tests on feature vs quality relationship.
    
    Args:
        feature_values: List of feature values
        quality_labels: List of quality labels
        good_threshold: Minimum quality label to consider "good"
        bad_threshold: Maximum quality label to consider "bad"
        
    Returns:
        Dictionary with test results (Spearman correlation, Mann-Whitney U)
    """
    try:
        from scipy.stats import spearmanr, mannwhitneyu
        
        if len(feature_values) != len(quality_labels):
            raise ValueError("feature_values and quality_labels must have same length")
        
        # Remove NaN values
        valid_pairs = [(f, q) for f, q in zip(feature_values, quality_labels) 
                      if np.isfinite(f)]
        
        if len(valid_pairs) < 3:
            return {'spearman_r': None, 'spearman_p': None, 
                   'mannwhitney_u': None, 'mannwhitney_p': None}
        
        features, qualities = zip(*valid_pairs)
        
        # Spearman rank correlation
        spearman_r, spearman_p = spearmanr(features, qualities)
        
        # Mann-Whitney U test (good vs bad groups)
        good_features = [f for f, q in valid_pairs if q >= good_threshold]
        bad_features = [f for f, q in valid_pairs if q <= bad_threshold]
        
        if len(good_features) < 1 or len(bad_features) < 1:
            mannwhitney_u, mannwhitney_p = None, None
        else:
            mannwhitney_u, mannwhitney_p = mannwhitneyu(good_features, bad_features)
        
        return {
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'mannwhitney_u': float(mannwhitney_u) if mannwhitney_u is not None else None,
            'mannwhitney_p': float(mannwhitney_p) if mannwhitney_p is not None else None,
            'n_good_samples': len(good_features),
            'n_bad_samples': len(bad_features),
            'n_valid_samples': len(valid_pairs)
        }
        
    except Exception:
        return {'spearman_r': None, 'spearman_p': None,
               'mannwhitney_u': None, 'mannwhitney_p': None}


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