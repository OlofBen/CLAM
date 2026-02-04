from __future__ import print_function

import argparse
import os
from multiprocessing import Manager

# internal imports
from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_MIL_Dataset

# pytorch imports
import torch
import torch.multiprocessing as mp

import pandas as pd
import numpy as np

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
args = parser.parse_args()

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

def main_worker(rank, args, shared_features, fold_queue, results_dict):
	os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
	# torch.cuda.set_device(rank)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Initialize skeleton dataset in this process
	worker_dataset = Generic_MIL_Dataset(
		csv_path = os.path.join('dataset_csv', '%s.csv' % args.task),
		data_dir = args.data_root_dir,
		shuffle = False, seed = args.seed, print_info = False, patient_strat = False
	)
	worker_dataset.set_shared_cache(shared_features)

	while not fold_queue.empty():
		try:
			i = fold_queue.get_nowait()
		except:
			break

		print(f"\n>>> [GPU {rank}] Starting Fold {i}...")
		seed_torch(args.seed)

		train_dataset, val_dataset, test_dataset = worker_dataset.return_splits(
			from_id=False, csv_path='{}/splits_{}.csv'.format(args.split_dir, i)
		)

		# Ensure splits use shared cache
		for split in [train_dataset, val_dataset, test_dataset]:
			if split is not None: split.features_cache = shared_features

		# Run training
		results, test_auc, val_auc, test_acc, val_acc = train((train_dataset, val_dataset, test_dataset), i, args)

		# Write individual pkl (Original behavior)
		filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
		save_pkl(filename, results)

		# Push metrics to shared results_dict for final summary.csv
		results_dict[i] = {
			'test_auc': test_auc,
			'val_auc': val_auc,
			'test_acc': test_acc,
			'val_acc': val_acc
		}
		print(f">>> [GPU {rank}] Fold {i} Complete. Test AUC: {test_auc:.4f}")

if __name__ == "__main__":
	try:
		mp.set_start_method('spawn', force=True)
	except RuntimeError:
		pass

	# 1. Setup Shared Memory
	manager = Manager()
	shared_features = manager.dict()
	fold_results = manager.dict() # To collect metrics from workers

	# 2. Preload Data (Once)
	print("Preloading data into Shared RAM...")
	master_dataset = Generic_MIL_Dataset(
		csv_path = os.path.join('dataset_csv', '%s.csv' % args.task),
		data_dir = args.data_root_dir, seed = args.seed
	)
	master_dataset.set_shared_cache(shared_features)
	master_dataset.preload_data()

	# 3. Setup Fold Queue
	# We use args.k_start/end to respect your original subset logic
	start = 0 if args.k_start == -1 else args.k_start
	end = args.k if args.k_end == -1 else args.k_end
	folds_to_run = np.arange(start, end)
	
	fold_queue = mp.Queue()
	for f in folds_to_run:
		fold_queue.put(f)

	# 4. Spawn Workers
	num_gpus = torch.cuda.device_count()
	processes = []
	for rank in range(num_gpus):
		p = mp.Process(target=main_worker, args=(rank, args, shared_features, fold_queue, fold_results))
		p.start()
		processes.append(p)

	for p in processes:
		p.join()

	# 5. RECONSTRUCT SUMMARY LOGGING
	print("\nTraining complete. Generating Summary CSV...")

	all_test_auc, all_val_auc, all_test_acc, all_val_acc = [], [], [], []

	# Extract results in the correct order
	for i in folds_to_run:
		if i in fold_results:
			m = fold_results[i]
			all_test_auc.append(m['test_auc'])
			all_val_auc.append(m['val_auc'])
			all_test_acc.append(m['test_acc'])
			all_val_acc.append(m['val_acc'])

	final_df = pd.DataFrame({
		'folds': folds_to_run, 
		'test_auc': all_test_auc, 
		'val_auc': all_val_auc, 
		'test_acc': all_test_acc, 
		'val_acc' : all_val_acc
	})

	save_name = 'summary.csv' if len(folds_to_run) == args.k else f'summary_partial_{start}_{end}.csv'
	final_df.to_csv(os.path.join(args.results_dir, save_name))
	
	print(f"Results saved to {args.results_dir}/{save_name}")
	print("end script")
