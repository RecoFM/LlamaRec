import os
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import wandb

from config import *
from model import *
from dataloader import *
from trainer import *

from pytorch_lightning import seed_everything

try:
    os.environ['WANDB_PROJECT'] = PROJECT_NAME
except:
    print('WANDB_PROJECT not available, please set it in config.py')


def main(args, export_root=None):
    seed_everything(args.seed)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = LRURec(args)
    if export_root == None:
        export_root = EXPERIMENT_ROOT + '/' + args.model_code + '/' + args.dataset_code
    
    trainer = LRUTrainer(args, model, train_loader, val_loader, test_loader, export_root, args.use_wandb)
    
    if not args.generate_only:
        trainer.train()
        trainer.test()
    else:
        print('Loading best model for candidate generation...')
        best_model_path = os.path.join(export_root, 'models', 'best_acc_model.pth')
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Best model checkpoint not found at {best_model_path}. Please train the model first.")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        best_model_dict = torch.load(best_model_path, map_location=device).get(STATE_DICT_KEY)
        model.load_state_dict(best_model_dict)
        print(f'Model loaded successfully on {device}')

    # Generate candidates for reranking
    retrieved_path = os.path.join(export_root, 'retrieved.pkl')
    if os.path.exists(retrieved_path) and args.generate_only and not args.force_generate:
        print(f'Retrieved candidates file already exists at {retrieved_path}')
        print('Use --force_generate to regenerate candidates')
    else:
        print('Generating candidates...')
        trainer.generate_candidates(retrieved_path)
        print(f'Candidates saved to {retrieved_path}')


if __name__ == "__main__":
    args.model_code = 'lru'
    set_template(args)
    main(args, export_root=None)

    # # searching best hyperparameters
    # for decay in [0, 0.01]:
    #     for dropout in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    #         args.weight_decay = decay
    #         args.bert_dropout = dropout
    #         args.bert_attn_dropout = dropout
    #         export_root = EXPERIMENT_ROOT + '/' + args.model_code + '/' + args.dataset_code + '/' + str(decay) + '_' + str(dropout)
    #         main(args, export_root=export_root)