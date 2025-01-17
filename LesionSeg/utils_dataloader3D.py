from utils_patchProcess_new import get_patchList_Train
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import nibabel as nib
import json


def patchExtract(img, gt, patchIdxAll, patchSelection, opts):
    img_pool, gt_pool = [], []
    
    for selectIdx in patchSelection:
        box = patchIdxAll[selectIdx]
        imS = img[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
        gtS = gt[box[0]:box[1], box[2]:box[3], box[4]:box[5]]

        if opts.AUG and (random.random()>0.8):
            angle = random.randint(-opts.rotate_train, opts.rotate_train)
            imS = ndimage.rotate(imS, angle, axes=(1, 2), reshape=False, mode='reflect')
            gtS = ndimage.rotate(gtS, angle, axes=(1, 2), reshape=False, mode='reflect')

        img_pool.append(imS), gt_pool.append(gtS)
    
    return img_pool, gt_pool


class TrainSet(Dataset):
    def __init__(self, opts=None):
        self.opts = opts

    def __getitem__(self, index):
        im_path = self.opts.pathAll[index] # imPath
        gt_path = im_path.replace('SUV', 'SEG') # gtPath
        box_path = im_path.replace(self.opts.imgDirSeg, self.opts.boundBoxDirSeg).replace('SUV.nii.gz', 'SUV.json')
        # bound box using utils_get_imgBoundBox.py
            
        im, gt = nib.load(im_path).get_fdata(), nib.load(gt_path).get_fdata() 
        assert im.shape == gt.shape, f"im.shape!=seg.shape, path: {im_path}"
        # ========crop==========================================================
        with open(box_path, 'r') as f:
            box = json.load(f)
        im = im[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
        gt = gt[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
        gt = np.where(gt > 0.5, 1.0, 0.0)
        
        # ========sample patch==============================================================
        patchIdxAll, tumorIndicatorAll = get_patchList_Train(img=im, patch_size=self.opts.patch_size, patch_moving_stride=self.opts.stride_size,
                                                             validValueThresh=self.opts.validValueThresh, segMask=gt) # np.sum(np.array(tumorIndicatorAll)>0.5)
        adaptiveSampleMetric = np.where(np.array(tumorIndicatorAll)>0.5, 10.0, 1.0)
        
        patchSelection = np.random.choice(np.arange(len(patchIdxAll)), size=self.opts.SamplePatchNumPerImage, 
                                        replace=False, p=adaptiveSampleMetric/(adaptiveSampleMetric.sum()+1e-20))
        # np.sum(np.array([tumorIndicatorAll[cc] for cc in patchSelection])>0.5)
        img_pool, gt_pool = patchExtract(img=im, gt=gt, patchIdxAll=patchIdxAll, patchSelection=patchSelection, opts=self.opts)
        vol_img, vol_gt = torch.from_numpy(np.stack(img_pool)), torch.from_numpy(np.stack(gt_pool))

        return {'vol_img': vol_img, 'vol_gt': vol_gt, 'vol_path': im_path}

    def __len__(self):
        return len(self.opts.pathAll)
