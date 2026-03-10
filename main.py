from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    start = 0 if args.k_start == -1 else args.k_start
    end = args.k if args.k_end == -1 else args.k_end

    # 1. Initialize metric lists
    # In survival mode, we use 'auc' lists to store C-Index for consistency
    all_test_auc, all_val_auc = [], []
    all_test_acc, all_val_acc = [], []
    all_test_f1, all_val_f1 = [], []
    all_test_precision, all_val_precision = [], []
    all_test_recall, all_val_recall = [], []

    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(args, from_id=False,
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))

        datasets = (train_dataset, val_dataset, test_dataset)

        # 2. Unpack metrics
        # The core_utils.train function now returns C-Index in the 3rd and 4th slots if survival=True
        (val_results, test_results, test_auc, val_auc, test_acc, val_acc,
         test_f1, val_f1, test_prec, val_prec, test_recall, val_recall) = train(datasets, i, args)

        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        all_test_f1.append(test_f1)
        all_val_f1.append(val_f1)
        all_test_precision.append(test_prec)
        all_val_precision.append(val_prec)
        all_test_recall.append(test_recall)
        all_val_recall.append(val_recall)

        combined_results = {'test': test_results, 'val': val_results}
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, combined_results)

    # 3. Create Summary DataFrame
    # If survival, rename 'auc' columns to 'c_index' for clarity in the CSV
    summary_columns = {
        'folds': folds,
        'test_auc' if not args.cox_survival else 'test_c_index': all_test_auc,
        'val_auc' if not args.cox_survival else 'val_c_index': all_val_auc,
    }

    if not args.cox_survival:
        summary_columns.update({
            'test_acc': all_test_acc, 'val_acc' : all_val_acc,
            'test_f1': all_test_f1, 'val_f1': all_val_f1,
            'test_precision': all_test_precision, 'val_precision': all_val_precision,
            'test_recall': all_test_recall, 'val_recall': all_val_recall,
        })

    final_df = pd.DataFrame(summary_columns)
    save_name = 'summary.csv' if len(folds) == args.k else 'summary_partial_{}_{}.csv'.format(start, end)
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use, '
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb',
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str)
parser.add_argument('--n_classes', type=int, default=2)
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False,
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
parser.add_argument('--bag_dropout', type=float, default=0.2, help='dropout over each bag of patches')
parser.add_argument('--accumilation_steps', type=int, default=1, help='backward iterations before optimizer step')
parser.add_argument('--cox_survival', action='store_true', default=False,
                    help='Enable Cox Survival Analysis mode')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

if args.cox_survival:
    args.n_classes = 1 # Cox head is always a single output node

encoding_size = args.embed_dim
settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt,
            'n_classes': args.n_classes}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Dataset')


dataset = Generic_MIL_Dataset(csv_path = os.path.join('dataset_csv', '%s.csv' % args.task),
                              data_dir= args.data_root_dir,
                              shuffle = False,
                              seed = args.seed,
                              print_info = True,
                              patient_strat=False,
                              bag_dropout=args.bag_dropout,
                              cox_survival=args.cox_survival)

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")


