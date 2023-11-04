"""
Summarize the DTU prediction metrics: LPIPS, PSNR, SSIM. 

This is after running python scripts/inference.py for the DTU checkpoints.
"""
from pathlib import Path 
import ipdb
import torch
import glob
from typing import List
import sys 
import numpy as np
import pandas as pd
from lpips import LPIPS

sys.path.append("..")
from training.inference_dtu import ssim_fn_batch, lpips_fn_batch, mse_to_psnr

lpips_fn = {} # gobal variable to hold a modle that we'll load

def compute_metrics(results: dict):
	""" """
	if len(lpips_fn) == 0:
		lpips_fn['lpips_fn'] = LPIPS(net="vgg").cuda()

	lpips_, ssim_, psnr_ = [], [], []
	imgs_gt = results['imgs_gt']
	masks = results['masks']
	assert imgs_gt.shape == masks.shape

	# iterate over the seeds 
	for i, imgs_pred in enumerate(results['imgs_pred']):
		ssim_batch = ssim_fn_batch(imgs_pred*masks, imgs_gt*masks)
		ssim_.append(ssim_batch.mean().item())

		lpips_batch = lpips_fn_batch(imgs_pred*masks, imgs_gt*masks, lpips_fn=lpips_fn['lpips_fn'])
		lpips_.append(lpips_batch.mean().item())

		bs = len(imgs_pred)
		mse_denominator  = masks.view(bs, -1).sum(dim=-1)
		mse_b = ((imgs_gt*masks - imgs_pred*masks)**2).view(bs, -1).sum(-1) / mse_denominator
		psnr_b = mse_to_psnr(mse_b) # (bs,num_seeds)
		psnr_.append(psnr_b.mean().item())

	metrics = torch.from_numpy(np.stack((lpips_, ssim_, psnr_)))   # shape (3,num_seeds) bc we have 3 metrics
	return metrics

def process_dtu_checkpoints():
	df_results = pd.DataFrame(columns=['num_imgs', 'dtu_subset', 'iteration','seed','lpips','ssim','psnr'])
	for dtu_subset in (1,3):
		for iteration in (1500,3000):	

			metrics_all = []
			dirs_results = glob.glob(f"results/20230805_scan*_subs_{dtu_subset}_m5_alpha5_augs7_pretrainkey8")
			
			for dir_results in dirs_results:
				results = torch.load(Path(dir_results) / f"inference/results_all_iter_{iteration}.pt")
				metrics_ = compute_metrics(results)
				metrics_all.append(metrics_)
			
			# compute the mean of the metrics over the scans from the dirs_results dirs
			metrics_all = torch.stack(metrics_all) # (n_scans,3,num_seeds) and n_scans==len(dirs_results)
			metrics_all = metrics_all.mean(0) # (3, num_seeds)

			# summarise the results into a row per seed, and add it to the results dir
			num_seeds = metrics_all.shape[1]
			df_results_this = pd.DataFrame(dict(
				num_imgs=[dtu_subset,]*num_seeds,
				dtu_subset=[dtu_subset,]*num_seeds,
				iteration=[iteration,]*num_seeds,
				seed=list(range(num_seeds)),
				lpips=metrics_all[0],
				ssim=metrics_all[1],
				psnr=metrics_all[2],
				))
			df_results = pd.concat((df_results, df_results_this))
			fname = 'results/summarize_dtu.csv'
			df_results.to_csv(fname)
			print(df_results)
			print()

	ipdb.set_trace()
	

if __name__=="__main__":
	# dir_inference = Path("results/20230805_scan103_subs_1_m5_alpha5_augs7_pretrainkey8/inference/")
	# iteration = 1500
	# read_images(dir_inference, iteration=iteration, )
	process_dtu_checkpoints()

