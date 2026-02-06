import time
import os
import argparse
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag
from models import get_encoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(loader, dataset_iter, model):
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""

	all_features = []
	for data in tqdm(dataset_iter, total=len(loader)):
		with torch.inference_mode():
			batch = data['img'].to(device, non_blocking=True)
			features = model(batch)

			all_features.append(features.cpu())

	return torch.cat(all_features, dim=0)

def fetch_dataset(bags_dataset, bag_candidate_idx, args, loader_kwargs, dest_files, img_transforms):
	slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
	bag_name = slide_id + '.h5'
	bag_candidate = os.path.join(args.data_dir, 'patches', bag_name)

	print(bag_name)
	if not os.path.exists(bag_candidate):
		print(f'Warning: {bag_name} not found at {bag_candidate}')
		return None, None, bag_name
	if not args.no_auto_skip and slide_id+'.pt' in dest_files:
		print('skipped {}'.format(slide_id))
		return None, None, bag_name

	file_path = bag_candidate

	dataset = Whole_Slide_Bag(file_path=file_path, img_transforms=img_transforms)
	loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
	dataset_iter = iter(loader)

	return loader, dataset_iter, bag_name

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--csv_path', type=str)
parser.add_argument('--feat_dir', type=str)
parser.add_argument('--model_name', type=str, default='resnet50_trunc')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224,
					help='the desired size of patches for scaling before feature embedding')
args = parser.parse_args()

if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)		
	model = model.to(device)

	if torch.cuda.device_count() > 1:
		print(f"Using {torch.cuda.device_count()} GPUs!")
		model = nn.DataParallel(model)

	model.eval()

	loader_kwargs = {'num_workers': 8,
					'prefetch_factor': 4, 
					'pin_memory': True,
					'persistent_workers': False,
					} if device.type == "cuda" else {}
	num_prefetch = 4

	total = len(bags_dataset)

	executor = ThreadPoolExecutor(max_workers=1)

	futures_queue = []
	for i in range(min(num_prefetch, total)):
		future = executor.submit(fetch_dataset, bags_dataset, i, args, loader_kwargs, dest_files, img_transforms)
		futures_queue.append(future)

	for bag_candidate_idx in range(total):
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))

		# Refill the queue
		next_job_idx = bag_candidate_idx + num_prefetch
		if next_job_idx < total:
			future = executor.submit(fetch_dataset, bags_dataset, next_job_idx, args, loader_kwargs, dest_files, img_transforms)
			futures_queue.append(future)

		try:
			current_future = futures_queue.pop(0)
			loader, dataset_iter, bag_name = current_future.result()
			print(f"\nProcessing file {bag_name}")
			if loader is None: 
				continue

			time_start = time.time()
			features = compute_w_loader(loader, dataset_iter, model, verbose=1)

			time_elapsed = time.time() - time_start

			bag_base, _ = os.path.splitext(bag_name)
			print('features size: ', features.shape)
			torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))

			del loader
			del dataset_iter

		except Exception as e:
			print(f"{bag_candidate_idx} got error {e}")

	executor.shutdown()
