import os
import glob
from collections import OrderedDict

import cv2
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
args = parser.parse_args()

test_results = OrderedDict()
test_results['psnr'] = []
test_results['ssim'] = []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
test_results['psnr_y'] = []                                                                       
test_results['ssim_y'] = []
test_results['psnr_b'] = []
psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0

datas = [ torch.load(f'./logits/{model}', map_location="cpu") for model in os.listdir('./logits')]

for idx, path in enumerate(sorted(glob.glob(os.path.join(args.folder_lq, '*')))):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    if args.folder_gt is not None:
        img_gt = (cv2.imread(args.folder_gt + '/' + imgname + imgext, cv2.IMREAD_COLOR).astype(np.float32) / 255. * 255.0).round().astype(np.uint8)
    
    img_sr = sum([data[imgname] for data in datas]) / len(datas)
    img_sr = (img_sr * 255.0).round().astype(np.uint8) 
    
    cv2.imwrite(f'./results/{imgname}.png', img_sr)

    if args.folder_gt is not None:
        test_results['psnr'].append(peak_signal_noise_ratio(img_gt, img_sr))
        test_results['ssim'].append(sum([structural_similarity(img_gt[:,:,item], img_sr[:,:,item]) for item in range(3)]) / 3.0)

if args.folder_gt is not None:
    print("Average PSNR/SSIM(RGB): {:.4f} dB; {:.4f}".format(sum(test_results["psnr"]) / len(test_results["psnr"]), sum(test_results["ssim"]) / len(test_results["ssim"])))

