# %%
# 1. Imports
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# %%
# 2. Core Loading and Parsing Functions
def load_thought_labels(filepath: Path) -> List[Dict[str, Any]]:
    """Loads and standardizes thought labels from a JSON file."""
    if not filepath.is_file():
        raise FileNotFoundError(f"Labels file not found: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            labels_list = json.load(f)

        if not isinstance(labels_list, list):
            raise ValueError(f"Expected list in {filepath}, got {type(labels_list)}")

        standardized_labels = []
        for i, item in enumerate(labels_list):
            if not isinstance(item, dict):
                continue
            standardized_item = {
                'thought_index': i,
                'input_text': item.get('Input', ''),
                'quality_label': item.get('label', item.get('Label', 0))
            }
            standardized_labels.append(standardized_item)

        print(f"✅ Successfully loaded {len(standardized_labels)} thought labels from {filepath.name}")
        return standardized_labels

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}")


def parse_circuit_filename(filename: str) -> Tuple[int, int]:
    """Parses circuit filename to extract thought and variation indices."""
    # New pattern: XX_thought_YY_variation.json
    pattern = r'(\d+)_thought_(\d+)_variation\.json'
    match = re.match(pattern, filename)
    if not match:
        # Try old pattern for backward compatibility
        old_pattern = r'(\d+)_th_thought_(\d+)_th_variation\.png\.json'
        match = re.match(old_pattern, filename)
        if not match:
            raise ValueError(f"Filename doesn't match expected pattern: {filename}")
    return int(match.group(1)), int(match.group(2))


def load_circuit_data(filepath: Path) -> Dict[str, Any]:
    """Loads circuit data from a single JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in circuit file {filepath}: {e}")


# %%
# 3. Data Unification
def create_unified_dataset(
    circuits_dir: Path,
    labels_list: List[Dict[str, Any]],
    dataset_name: str = "unknown"
) -> List[Dict[str, Any]]:
    """Creates a unified dataset by matching circuit files with thought labels.
    
    Each circuit file (XX_thought_YY_variation.json) maps to one label entry:
    - The combination of thought_index and variation_index determines which label to use
    - Formula: label_index = thought_index * variations_per_thought + variation_index
    """
    all_records = []
    missing_circuits = []
    
    # Get all available circuit files
    circuit_files = list(circuits_dir.glob("*_thought_*_variation.json"))
    
    # Also check for old naming pattern
    if not circuit_files:
        circuit_files = list(circuits_dir.glob("*_th_thought_*_variation.png.json"))
    
    if not circuit_files:
        print(f"No circuit files found in {circuits_dir}")
        return []
    
    print(f"Found {len(circuit_files)} circuit files")
    
    # Parse all circuit files and map to labels
    for circuit_filepath in sorted(circuit_files):
        try:
            thought_idx, variation_idx = parse_circuit_filename(circuit_filepath.name)
            
            # Calculate which label entry this corresponds to
            # Assuming 5 variations per thought (00-04), but we can determine this dynamically
            variations_per_thought = 5  # This could be made configurable
            label_index = thought_idx * variations_per_thought + variation_idx
            
            if label_index < len(labels_list):
                # Get the corresponding label
                label_data = labels_list[label_index].copy()
                label_data['thought_index'] = label_index  # Update to match the actual index
                
                # Load circuit data
                circuit_data = load_circuit_data(circuit_filepath)
                
                record = {
                    'dataset': dataset_name,
                    'thought_index': label_index,
                    'original_thought_idx': thought_idx,
                    'variation_index': variation_idx,
                    'filename': circuit_filepath.name,
                    **label_data,
                    'circuit_data': circuit_data,
                }
                all_records.append(record)
            else:
                missing_circuits.append(f"No label for {circuit_filepath.name} (calculated index {label_index})")
                
        except (ValueError, json.JSONDecodeError) as e:
            missing_circuits.append(f"Failed to load {circuit_filepath.name}: {e}")
    
    # Add records for labels without circuit data
    circuit_label_indices = {r['thought_index'] for r in all_records}
    for i, label_data in enumerate(labels_list):
        if i not in circuit_label_indices:
            label_data_copy = label_data.copy()
            label_data_copy['thought_index'] = i
            
            record = {
                'dataset': dataset_name,
                'thought_index': i,
                'original_thought_idx': None,
                'variation_index': None,
                'filename': None,
                **label_data_copy,
                'circuit_data': None,
            }
            all_records.append(record)

    print(f"\n--- 📊 {dataset_name.upper()} Dataset Loading Stats ---")
    print(f"Total records created: {len(all_records)}")
    print(f"Records with circuit data: {len([r for r in all_records if r['circuit_data'] is not None])}")
    print(f"Records without circuit data: {len([r for r in all_records if r['circuit_data'] is None])}")
    if missing_circuits:
        print(f"Missing/failed circuits: {len(missing_circuits)}")
        for failure in missing_circuits[:3]:
            print(f"  - {failure}")
    print("-" * 50)
    return all_records


# %%
# 4. Main Public API Function
def load_and_build_dataframe(dataset_name: str, base_path: str = "data") -> pd.DataFrame:
    """Loads a complete dataset and returns it as a pandas DataFrame."""
    base_dir = Path(base_path)
    
    # Extract suffix from dataset name (e.g., "test2" -> "2", "test3" -> "3")
    label_suffix = dataset_name.replace('test', '').lstrip('_')
    labels_path = base_dir / f"labels{label_suffix}.json"

    print(f"\n{'='*50}")
    print(f"🚀 LOADING {dataset_name.upper()} DATASET")
    print(f"Looking for labels file: {labels_path}")
    print(f"{'='*50}")

    labels_list = load_thought_labels(labels_path)
    unified_records = create_unified_dataset(base_dir / "raw" / dataset_name, labels_list, dataset_name)

    if not unified_records:
        print("⚠️ No records were loaded, returning an empty DataFrame.")
        return pd.DataFrame()
        
    df = pd.DataFrame(unified_records)
    print(f"✅ DataFrame created successfully with {len(df)} rows.")
    return df


# %%
# 5. Example Usage (for direct execution)
if __name__ == '__main__':
    print("Loading test2 dataset for demonstration...")
    
    try:
        df_test2 = load_and_build_dataframe("test2", base_path="data")
        if not df_test2.empty:
            print("\n--- DataFrame Info ---")
            df_test2.info()
            print("\n--- DataFrame Head ---")
            print(df_test2.head())
            print(f"\n--- Quality Label Distribution ---")
            print(df_test2['quality_label'].value_counts().sort_index())
        else:
            print("No data loaded.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")