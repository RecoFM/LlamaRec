import pandas as pd
import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "repo_id": "ChernovAndrei/reco-fm-data",
    "subset": "5core_last_out_w_his_All_Beauty",
    "output_dir": "../data/preprocessed/small_beauty_2023_min_rating0-min_uc5-min_sc5",
    "output_filename": "dataset.pkl"
}

def setup_directories(output_dir: str) -> None:
    """Create output directories if they don't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

def load_split_data(api: HfApi, repo_id: str, file_pattern: str) -> pd.DataFrame:
    """Load a specific data split from HuggingFace Hub."""
    repo_files = api.list_repo_files(repo_id, repo_type="dataset")
    matching_files = [f for f in repo_files if file_pattern in f]
    
    if not matching_files:
        raise ValueError(f"No files found matching pattern: {file_pattern}")
    
    dataset = load_dataset(repo_id, data_files=matching_files[0], split='train')
    return dataset.to_pandas()

def create_id_mappings(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Create user and item ID mappings."""
    # Combine unique IDs from all splits
    all_users = pd.concat([
        train_df['user_id'],
        val_df['user_id'],
        test_df['user_id']
    ]).unique()
    
    all_items = pd.concat([
        train_df['parent_asin'],
        val_df['parent_asin'],
        test_df['parent_asin']
    ]).unique()
    
    # Create mappings starting from 1
    user_map = {user_id: idx + 1 for idx, user_id in enumerate(all_users)}
    item_map = {item_id: idx + 1 for idx, item_id in enumerate(all_items)}
    
    logger.info(f"Created mappings for {len(user_map)} users and {len(item_map)} items")
    return user_map, item_map

def map_dataframe(df: pd.DataFrame, user_map: Dict[str, int], item_map: Dict[str, int]) -> pd.DataFrame:
    """Map user_ids and item_ids in a dataframe to their integer representations."""
    df_mapped = df.copy()
    df_mapped['uid'] = df_mapped['user_id'].map(user_map)
    df_mapped['sid'] = df_mapped['parent_asin'].map(item_map)
    
    # Map history items if present
    if 'history' in df_mapped.columns:
        df_mapped['history'] = df_mapped['history'].apply(
            lambda x: ' '.join([str(item_map[item]) for item in x.split()]) if isinstance(x, str) else ''
        )
    
    return df_mapped

def create_interaction_dict(df: pd.DataFrame, use_history: bool = False) -> Dict[int, List[int]]:
    """Create a dictionary of user interactions."""
    if use_history:
        interactions = {
            row['uid']: [int(x) for x in row['history'].split()] if row['history'] else []
            for _, row in df.iterrows()
        }
    else:
        interactions = {
            row['uid']: [row['sid']] 
            for _, row in df.iterrows()
        }
    return dict(sorted(interactions.items()))

def create_metadata_dict(meta_df: pd.DataFrame, item_map: Dict[str, int]) -> Dict[int, str]:
    """Create a dictionary of item metadata."""
    meta_df = meta_df.copy()
    meta_df['sid'] = meta_df['parent_asin'].map(item_map)
    metadata = {row['sid']: row['title'] for _, row in meta_df.iterrows()}
    return dict(sorted(metadata.items()))

def perform_sanity_checks(final_dataset: Dict[str, Any]) -> None:
    """Perform sanity checks on the processed dataset."""
    # Check for data presence
    for split in ['train', 'val', 'test']:
        n_users = len(final_dataset[split])
        n_interactions = sum(len(items) for items in final_dataset[split].values())
        logger.info(f"{split} split: {n_users} users, {n_interactions} interactions")
    
    # Check mappings
    n_users = len(final_dataset['umap'])
    n_items = len(final_dataset['smap'])
    logger.info(f"Total unique users: {n_users}, items: {n_items}")
    
    # Check metadata coverage
    n_items_with_meta = len(final_dataset['meta'])
    coverage = n_items_with_meta / n_items * 100
    logger.info(f"Metadata coverage: {coverage:.2f}% ({n_items_with_meta}/{n_items} items)")
    
    # Check ID ranges
    all_uids = set()
    all_sids = set()
    for split in ['train', 'val', 'test']:
        all_uids.update(final_dataset[split].keys())
        all_sids.update([sid for items in final_dataset[split].values() for sid in items])
    
    logger.info(f"User ID range: {min(all_uids)} to {max(all_uids)}")
    logger.info(f"Item ID range: {min(all_sids)} to {max(all_sids)}")

def main():
    """Main preprocessing pipeline."""
    logger.info("Starting data preprocessing...")
    
    # Setup
    setup_directories(CONFIG['output_dir'])
    api = HfApi()
    
    # Load data
    logger.info("Loading data from HuggingFace Hub...")
    train_df = load_split_data(api, CONFIG['repo_id'], "train/train_data.parquet")
    val_df = load_split_data(api, CONFIG['repo_id'], "val/val_data.parquet")
    test_df = load_split_data(api, CONFIG['repo_id'], "test/test_data.parquet")
    meta_df = load_split_data(api, CONFIG['repo_id'], "embeddings/title_embeddings.parquet")
    
    # Create ID mappings
    user_map, item_map = create_id_mappings(train_df, val_df, test_df)
    
    # Process each split
    df_val_mapped = map_dataframe(val_df, user_map, item_map)
    df_test_mapped = map_dataframe(test_df, user_map, item_map)
    
    # Create final dataset structure
    final_dataset = {
        'train': create_interaction_dict(df_val_mapped, use_history=True),
        'val': create_interaction_dict(df_val_mapped),
        'test': create_interaction_dict(df_test_mapped),
        'meta': create_metadata_dict(meta_df.drop(['embedding'], axis=1), item_map),
        'smap': item_map,
        'umap': user_map
    }
    
    # Perform sanity checks
    perform_sanity_checks(final_dataset)
    
    # Save the processed dataset
    output_path = Path(CONFIG['output_dir']) / CONFIG['output_filename']
    with open(output_path, 'wb') as f:
        pickle.dump(final_dataset, f)
    logger.info(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    main()

