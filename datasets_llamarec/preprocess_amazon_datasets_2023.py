import pandas as pd
import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_config(dataset_name: str) -> Dict[str, str]:
    """Get configuration based on dataset name."""
    return {
        "repo_id": "ChernovAndrei/reco-fm-data",
        "subset": f"5core_last_out_w_his_{dataset_name}",
        "output_dir": f"../data/preprocessed/{dataset_name}_2023_min_rating0-min_uc5-min_sc5",
        "output_filename": "dataset.pkl"
    }

def setup_directories(output_dir: str) -> None:
    """Create output directories if they don't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

def load_split_data(api: HfApi, repo_id: str, subset: str, file_pattern: str) -> pd.DataFrame:
    """Load a specific data split from HuggingFace Hub.
    
    Args:
        api: HuggingFace API instance
        repo_id: Repository ID on HuggingFace
        subset: Dataset subset name (e.g., '5core_last_out_w_his_All_Beauty')
        file_pattern: Pattern to match the file name
    """
    repo_files = api.list_repo_files(repo_id, repo_type="dataset")
    matching_files = [f for f in repo_files if subset in f and file_pattern in f]
    
    if not matching_files:
        raise ValueError(f"No files found matching subset '{subset}' and pattern: {file_pattern}")
    
    dataset = load_dataset(repo_id, data_files=matching_files[0], split='train')
    return dataset.to_pandas()

def check_for_nans(df: pd.DataFrame, split_name: str) -> None:
    """Check for NaN values in critical columns and raise error if found.
    
    For training set, NaN values are allowed in history column.
    For validation and test sets, no NaN values are allowed in any column.
    """
    # These columns should never have NaN values in any split
    critical_columns = {
        'user_id': df['user_id'].isna().sum(),
        'parent_asin': df['parent_asin'].isna().sum(),
    }
    
    if any(critical_columns.values()):
        error_msg = f"Found NaN values in {split_name} split critical columns:\n"
        for col, count in critical_columns.items():
            if count > 0:
                error_msg += f"- {col}: {count} NaN values\n"
                error_msg += f"Example rows with NaN in {col}:\n"
                error_msg += df[df[col].isna()].head().to_string() + "\n"
        raise ValueError(error_msg)
    
    # Check history column based on split type
    if 'history' in df.columns:
        history_nans = df['history'].isna().sum()
        if history_nans > 0 and split_name.lower() != "train":
            error_msg = f"Found {history_nans} NaN values in history column of {split_name} split.\n"
            error_msg += "NaN values in history are only allowed in training set.\n"
            error_msg += "Example rows with NaN history:\n"
            error_msg += df[df['history'].isna()].head().to_string()
            raise ValueError(error_msg)
        elif history_nans > 0:
            logger.info(f"Found {history_nans} NaN values in history column of training set (this is allowed)")

def create_id_mappings(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Create user and item ID mappings from all splits, but history only from validation."""
    # Get all users from all splits
    all_users = pd.concat([
        train_df['user_id'],
        val_df['user_id'],
        test_df['user_id']
    ]).unique()
    
    # Get items from direct interactions from all splits
    items_from_interactions = pd.concat([
        train_df['parent_asin'],
        val_df['parent_asin'],
        test_df['parent_asin']
    ]).unique()
    
    # Get items from history in validation set only
    items_from_history = set()
    if 'history' in val_df.columns:
        history_series = val_df['history'].dropna()
        history_items = history_series.str.split().explode().unique()
        items_from_history.update(history_items)
    
    # Combine all unique items
    all_items = sorted(set(items_from_interactions) | items_from_history)
    
    # Create mappings starting from 1
    user_map = {str(user_id): idx + 1 for idx, user_id in enumerate(all_users)}
    item_map = {str(item_id): idx + 1 for idx, item_id in enumerate(all_items)}
    
    logger.info(f"Created mappings for {len(user_map)} users and {len(item_map)} items")
    logger.info(f"Items from direct interactions: {len(items_from_interactions)}")
    logger.info(f"Additional items from history: {len(items_from_history - set(items_from_interactions))}")
    
    return user_map, item_map

def map_dataframe(df: pd.DataFrame, user_map: Dict[str, int], item_map: Dict[str, int]) -> pd.DataFrame:
    """Map user_ids and item_ids in a dataframe to their integer representations."""
    df_mapped = df.copy()
    df_mapped['uid'] = df_mapped['user_id'].map(user_map)
    df_mapped['sid'] = df_mapped['parent_asin'].map(item_map)
    
    # Map history items if present
    if 'history' in df_mapped.columns:
        df_mapped['history'] = df_mapped['history'].apply(
            lambda x: ' '.join([str(item_map.get(item, 0)) for item in x.split()]) if isinstance(x, str) else ''
        )
    
    return df_mapped

def create_interaction_dict(df: pd.DataFrame, use_history: bool = False) -> Dict[int, List[int]]:
    """Create a dictionary of user interactions.
    
    Args:
        df: DataFrame containing user interactions
        use_history: If True, use the history column for interactions (for training)
                    If False, use the target item (for val/test)
    """
    interactions = {}
    
    for _, row in df.iterrows():
        uid = row['uid']
        if use_history:
            # For training data, use only history
            if pd.isna(row['history']):
                raise ValueError(f"User {uid} has no history but history mode is enabled")
            else:
                # Split history string and convert to integers
                interactions[uid] = [int(x) for x in row['history'].split()]
        else:
            # For val/test data, use only the target item
            interactions[uid] = [row['sid']]
    
    # Log some statistics
    if use_history:
        total_history_items = sum(len(items) for items in interactions.values())
        logger.info(f"Created training dict from history: {len(interactions)} users, {total_history_items} total interactions")
    else:
        logger.info(f"Created target dict: {len(interactions)} users, {len(interactions)} interactions")
    
    return dict(sorted(interactions.items()))

def filter_and_validate_users(train_dict: Dict[int, List[int]], 
                            val_dict: Dict[int, List[int]], 
                            test_dict: Dict[int, List[int]]) -> Set[int]:
    """Filter users to keep only those present in all splits and log the filtering stats."""
    train_users = set(train_dict.keys())
    val_users = set(val_dict.keys())
    test_users = set(test_dict.keys())
    
    # Find users present in all splits
    complete_users = train_users & val_users & test_users
    
    # Log filtering statistics
    logger.info(f"User filtering statistics:")
    logger.info(f"- Initial users in train: {len(train_users)}")
    logger.info(f"- Initial users in val: {len(val_users)}")
    logger.info(f"- Initial users in test: {len(test_users)}")
    logger.info(f"- Users with complete data (train+val+test): {len(complete_users)}")
    logger.info(f"- Removed users: {len(train_users | val_users | test_users) - len(complete_users)}")
    
    if not complete_users:
        raise ValueError("No users have data in all splits!")
        
    return complete_users

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Preprocess Amazon dataset for recommendation.')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset (e.g., All_Beauty, Sports_and_Outdoors, Toys_and_Games)')
    args = parser.parse_args()
    
    # Get configuration based on dataset name
    CONFIG = get_config(args.dataset_name)
    
    logger.info("Starting data preprocessing...")
    logger.info(f"Processing dataset: {args.dataset_name}")
    logger.info(f"Processing subset: {CONFIG['subset']}")
    
    # Setup
    setup_directories(CONFIG['output_dir'])
    api = HfApi()
    
    # Load data
    logger.info("Loading data from HuggingFace Hub...")
    train_df = load_split_data(api, CONFIG['repo_id'], CONFIG['subset'], "train/train_data.parquet")
    val_df = load_split_data(api, CONFIG['repo_id'], CONFIG['subset'], "val/val_data.parquet")
    test_df = load_split_data(api, CONFIG['repo_id'], CONFIG['subset'], "test/test_data.parquet")
    meta_df = load_split_data(api, CONFIG['repo_id'], CONFIG['subset'], "embeddings/title_embeddings.parquet")
    
    # Create ID mappings using all splits to ensure complete coverage
    user_map, item_map = create_id_mappings(train_df, val_df, test_df)
    
    # Process validation and test splits
    df_val_mapped = map_dataframe(val_df, user_map, item_map)
    df_test_mapped = map_dataframe(test_df, user_map, item_map)
    
    # Create interaction dictionaries
    # Training data comes from validation history
    train_dict = create_interaction_dict(df_val_mapped, use_history=True)
    # Validation and test data are the target items
    val_dict = create_interaction_dict(df_val_mapped)
    test_dict = create_interaction_dict(df_test_mapped)
    
    # Filter to users with complete data
    complete_users = filter_and_validate_users(train_dict, val_dict, test_dict)
    
    # Filter dictionaries to keep only complete users
    train_dict = {uid: items for uid, items in train_dict.items() if uid in complete_users}
    val_dict = {uid: items for uid, items in val_dict.items() if uid in complete_users}
    test_dict = {uid: items for uid, items in test_dict.items() if uid in complete_users}
    
    # Create metadata dictionary
    meta_dict = create_metadata_dict(meta_df, item_map)
    
    # Create final dataset dictionary
    final_dataset = {
        'train': train_dict,
        'val': val_dict,
        'test': test_dict,
        'umap': {v: k for k, v in user_map.items()},  # Reverse mapping for convenience
        'smap': {v: k for k, v in item_map.items()},  # Reverse mapping for convenience
        'meta': meta_dict
    }
    
    # Perform sanity checks
    perform_sanity_checks(final_dataset)
    
    # Save processed dataset
    output_path = Path(CONFIG['output_dir']) / CONFIG['output_filename']
    with open(output_path, 'wb') as f:
        pickle.dump(final_dataset, f)
    logger.info(f"Saved processed dataset to {output_path}")

if __name__ == "__main__":
    main()

