from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from .utils import *
from .loggers import *
from .base import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import json
import pickle
import numpy as np
from abc import *
from pathlib import Path
import os


class LRUTrainer(BaseTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root, use_wandb):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root, use_wandb)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
    
    def calculate_loss(self, batch):
        seqs, labels = batch
        logits = self.model(seqs)
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch, exclude_history=True):
        seqs, labels = batch
        
        scores = self.model(seqs)[:, -1, :]
        B, L = seqs.shape
        if exclude_history:
            for i in range(L):
                scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
            scores[:, 0] = -1e9  # padding
        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels.view(-1), self.metric_ks)
        return metrics
    
    def generate_candidates(self, retrieved_data_path):
        self.model.eval()
        
        # Create temporary files for incremental saving
        val_tmp_file = retrieved_data_path + '.val.tmp'
        test_tmp_file = retrieved_data_path + '.test.tmp'
        
        with torch.no_grad():
            print('*************** Generating Candidates for Validation Set ***************')
            val_probs, val_labels = [], []
            chunk_size = 1000  # Save every 1000 samples
            
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = self.to_device(batch)
                seqs, labels = batch
        
                scores = self.model(seqs)[:, -1, :]
                B, L = seqs.shape
                for i in range(L):
                    scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
                scores[:, 0] = -1e9  # padding
                
                val_probs.extend(scores.cpu().tolist())
                val_labels.extend(labels.view(-1).cpu().tolist())
                
                # Save chunks to disk and clear memory
                if len(val_probs) >= chunk_size:
                    with open(val_tmp_file, 'ab') as f:
                        pickle.dump({'probs': val_probs, 'labels': val_labels}, f)
                    val_probs, val_labels = [], []  # Clear memory
            
            # Save any remaining validation data
            if val_probs:
                with open(val_tmp_file, 'ab') as f:
                    pickle.dump({'probs': val_probs, 'labels': val_labels}, f)
            
            # Load and combine validation chunks for metrics
            val_all_probs, val_all_labels = [], []
            with open(val_tmp_file, 'rb') as f:
                while True:
                    try:
                        chunk = pickle.load(f)
                        val_all_probs.extend(chunk['probs'])
                        val_all_labels.extend(chunk['labels'])
                    except EOFError:
                        break
            
            val_metrics = absolute_recall_mrr_ndcg_for_ks(
                torch.tensor(val_all_probs), 
                torch.tensor(val_all_labels).view(-1),
                self.metric_ks
            )
            print(val_metrics)
            
            print('****************** Generating Candidates for Test Set ******************')
            test_probs, test_labels = [], []
            
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = self.to_device(batch)
                seqs, labels = batch
        
                scores = self.model(seqs)[:, -1, :]
                B, L = seqs.shape
                for i in range(L):
                    scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
                scores[:, 0] = -1e9  # padding
                
                test_probs.extend(scores.cpu().tolist())
                test_labels.extend(labels.view(-1).cpu().tolist())
                
                # Save chunks to disk and clear memory
                if len(test_probs) >= chunk_size:
                    with open(test_tmp_file, 'ab') as f:
                        pickle.dump({'probs': test_probs, 'labels': test_labels}, f)
                    test_probs, test_labels = [], []  # Clear memory
            
            # Save any remaining test data
            if test_probs:
                with open(test_tmp_file, 'ab') as f:
                    pickle.dump({'probs': test_probs, 'labels': test_labels}, f)
            
            # Load and combine test chunks for metrics
            test_all_probs, test_all_labels = [], []
            with open(test_tmp_file, 'rb') as f:
                while True:
                    try:
                        chunk = pickle.load(f)
                        test_all_probs.extend(chunk['probs'])
                        test_all_labels.extend(chunk['labels'])
                    except EOFError:
                        break
            
            test_metrics = absolute_recall_mrr_ndcg_for_ks(
                torch.tensor(test_all_probs),
                torch.tensor(test_all_labels).view(-1),
                self.metric_ks
            )
            print(test_metrics)

        # Combine all data into final file
        with open(retrieved_data_path, 'wb') as f:
            pickle.dump({
                'val_probs': val_all_probs,
                'val_labels': val_all_labels,
                'val_metrics': val_metrics,
                'test_probs': test_all_probs,
                'test_labels': test_all_labels,
                'test_metrics': test_metrics
            }, f)
        
        # Clean up temporary files
        os.remove(val_tmp_file)
        os.remove(test_tmp_file)