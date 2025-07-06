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
        val_probs, val_labels = [], []
        test_probs, test_labels = [], []
        TOP_K = 100  # Number of candidates to keep
        
        with torch.no_grad():
            print('*************** Generating Candidates for Validation Set ***************')
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = self.to_device(batch)
                seqs, labels = batch
        
                scores = self.model(seqs)[:, -1, :]
                B, L = seqs.shape
                for i in range(L):
                    scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
                scores[:, 0] = -1e9  # padding
                
                # Get top k scores and indices
                top_scores, _ = torch.topk(scores, k=min(TOP_K, scores.size(1)), dim=1)
                # Convert to sparse format - only store top k scores
                val_probs.extend(top_scores.tolist())
                val_labels.extend(labels.view(-1).tolist())
            
            # Calculate metrics using full scores for accurate evaluation
            val_metrics = absolute_recall_mrr_ndcg_for_ks(scores, 
                                                          torch.tensor(val_labels).view(-1), self.metric_ks)
            print(val_metrics)

            print('****************** Generating Candidates for Test Set ******************')
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = self.to_device(batch)
                seqs, labels = batch
        
                scores = self.model(seqs)[:, -1, :]
                B, L = seqs.shape
                for i in range(L):
                    scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
                scores[:, 0] = -1e9  # padding
                
                # Get top k scores and indices
                top_scores, _ = torch.topk(scores, k=min(TOP_K, scores.size(1)), dim=1)
                # Convert to sparse format - only store top k scores
                test_probs.extend(top_scores.tolist())
                test_labels.extend(labels.view(-1).tolist())
            
            # Calculate metrics using full scores for accurate evaluation
            test_metrics = absolute_recall_mrr_ndcg_for_ks(scores,
                                                           torch.tensor(test_labels).view(-1), self.metric_ks)
            print(test_metrics)

        with open(retrieved_data_path, 'wb') as f:
            pickle.dump({'val_probs': val_probs,
                         'val_labels': val_labels,
                         'val_metrics': val_metrics,
                         'test_probs': test_probs,
                         'test_labels': test_labels,
                         'test_metrics': test_metrics}, f)