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
    pattern = r'(\d+)_th_thought_(\d+)_th_variation\.png\.json'
    match = re.match(pattern, filename)
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
    """Creates a unified dataset by matching circuit files with thought labels."""
    all_records = []
    missing_circuits = []
    
    # Start with all labels and find corresponding circuits
    for label_data in labels_list:
        thought_idx = label_data['thought_index']
        
        # Look for circuit files for this thought
        circuit_pattern = f"{thought_idx}_th_thought_*_variation.png.json"
        matching_circuits = list(circuits_dir.glob(circuit_pattern))
        
        if not matching_circuits:
            # No circuit files for this thought - create record with None circuit data
            record = {
                'dataset': dataset_name,
                'thought_index': thought_idx,
                'variation_index': None,
                'filename': None,
                **label_data,
                'circuit_data': None,
            }
            all_records.append(record)
            missing_circuits.append(f"No circuit files for thought {thought_idx}")
        else:
            # Create ONE record per thought using the first available circuit
            circuit_filepath = matching_circuits[0]  # Use first circuit variation
            try:
                thought_idx_parsed, variation_idx = parse_circuit_filename(circuit_filepath.name)
                circuit_data = load_circuit_data(circuit_filepath)
                
                record = {
                    'dataset': dataset_name,
                    'thought_index': thought_idx,
                    'variation_index': variation_idx,
                    'filename': circuit_filepath.name,
                    'available_variations': len(matching_circuits),
                    **label_data,
                    'circuit_data': circuit_data,
                }
                all_records.append(record)
            except (ValueError, json.JSONDecodeError) as e:
                # If first circuit fails, create record with None data
                record = {
                    'dataset': dataset_name,
                    'thought_index': thought_idx,
                    'variation_index': None,
                    'filename': None,
                    'available_variations': len(matching_circuits),
                    **label_data,
                    'circuit_data': None,
                }
                all_records.append(record)
                missing_circuits.append(f"Failed to load {circuit_filepath.name}: {e}")

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