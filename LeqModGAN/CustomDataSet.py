import random
import numpy as np
import torch
from torch.utils.data import Dataset
import math
from scipy import ndimage
import matplotlib.pyplot as plt
import nibabel as nib
import json

def get_patchList_Train(img, patch_size, patch_moving_stride, validValueThresh=0.15, segMask=None):
    img_size = img.shape
    imax, jmax, kmax = img_size[1] - patch_size[1], img_size[2] - patch_size[2], img_size[0] - patch_size[0]
    irange = list(range(0, imax+1, patch_moving_stride[1]))
    jrange = list(range(0, jmax+1, patch_moving_stride[2]))
    krange = list(range(0, kmax+1, patch_moving_stride[0]))
    if irange[-1] != imax:
        irange.append(imax)
    if jrange[-1] != jmax:
        jrange.append(jmax)
    if krange[-1] != kmax:
        krange.append(kmax)

    patchIdx, tumorIndicator = [], []
    for k in krange:
        for j in jrange:
            for i in irange:
                box = [k, k+patch_size[0], i, i+patch_size[1], j, j+patch_size[2]]
                
                imC = img[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
                numValidVoxels = np.sum(imC > validValueThresh)
                numAllvoxels = np.sum(imC >= 0.0)
                if numValidVoxels/numAllvoxels > 0.2: # filter out patches with a lot of zero voxels
                    patchIdx.append(box)

                    if segMask is not None:
                        segC = segMask[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
                        tumorIndicator.append(segC.max()) # patch lesion probability
    
    return patchIdx, tumorIndicator

def patchExtract(gt, im, patchIdxAll, patchSelection, opts, segMask=None, im_path=None):
    low_pool, high_pool = [], []
    lossWeight_pool = []
    for selectIdx in patchSelection:
        box = patchIdxAll[selectIdx]
        highS = gt[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
        lowS = im[box[0]:box[1], box[2]:box[3], box[4]:box[5]]

        if opts.weightLoss_lesionSUVbias is not None:
            tumorS = segMask[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
        else:
            tumorS = None

        if opts.AUG and (random.random()>0.8):
            angle = random.randint(-opts.rotate_train, opts.rotate_train)
            lowS = ndimage.rotate(lowS, angle, axes=(1, 2), reshape=False, mode='reflect')
            highS = ndimage.rotate(highS, angle, axes=(1, 2), reshape=False, mode='reflect')
            if tumorS is not None:
                tumorS = ndimage.rotate(tumorS, angle, axes=(1, 2), reshape=False, mode='reflect')

        if opts.weightLoss_lesionSUVbias is not None:
            weightS = tumorS
        else:
            weightS = None

        low_pool.append(lowS), high_pool.append(highS)
        
        if weightS is not None:
            lossWeight_pool.append(np.float32(weightS))
    return high_pool, low_pool, lossWeight_pool


class TrainSet(Dataset):
    def __init__(self, opts=None):
        self.opts = opts

    def __getitem__(self, index):
        im_path = self.opts.pathAll[index] # low-count image path
        gt_path = im_path.replace(im_path.split('_')[-1], '100.nii.gz') # high-count image path
        box_path = gt_path.replace('00._datasetDenoise/', '00._datasetDenoise_boundBox/').replace('.nii.gz', '.json') 
        # boundBox path pre-obtained by the file 'utils_get_imgBoundBox.py'
        tumor_path = gt_path.replace('00._datasetDenoise/', '00._datasetDenoise_boundBox_seg/')
        # lesion segmentation probability pre-obtained by the dir 'lesionSeg/'
        
        # ====================load img and crop==========================================================
        im, gt = nib.load(im_path).get_fdata(), nib.load(gt_path).get_fdata()
        segMask = nib.load(tumor_path).get_fdata()
        assert im.shape == gt.shape, f"lowIm.shape!=highIm.shape, path: {im_path}"
        with open(box_path, 'r') as f:
            box = json.load(f)
        im = im[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
        gt = gt[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
        segMask = segMask[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
        im[im<=0.0], gt[gt<=0.0] = 0.0, 0.0
        # plt.imsave('c1Fig.png',im[im.shape[0]//2,:,:], cmap='jet', vmin=0.03, vmax=5.0)
        # plt.imsave('c2Fig.png',gt[gt.shape[0]//2,:,:], cmap='jet', vmin=0.03, vmax=5.0)
        # plt.imsave('c3Fig.png',segMask[segMask.shape[0]//2,:,:], cmap='gray', vmin=0, vmax=1.0)
        
        # ======================divide img into patches==============================================================
        assert im.shape[0] >= self.opts.patch_size[0] and im.shape[1] >= self.opts.patch_size[1] and im.shape[2] >= self.opts.patch_size[2], \
            f"imshape after cropped smaller than patch shape: {im_path}"
        patchIdxAll, tumorIndicatorAll = get_patchList_Train(img=gt, patch_size=self.opts.patch_size, patch_moving_stride=self.opts.stride_size, 
                                                             validValueThresh=self.opts.validValueThresh, segMask=segMask)
        # index of patches and the patch lesion probability
        
        # =========================lesion-perceived sampling==============================================================
        assert len(patchIdxAll) >= self.opts.SamplePatchNumPerImage, f"numPatches smaller than numSamples: {im_path}"
        
        if self.opts.adaptSampleByLesion:
            adaptiveSampleMetric = np.array(tumorIndicatorAll) 
            adaptiveSampleMetric[adaptiveSampleMetric<self.opts.segProbThresh] = self.opts.segProbThresh 
            # set the minimum sampling weight
            patchSelection = np.random.choice(np.arange(len(patchIdxAll)), size=self.opts.SamplePatchNumPerImage, replace=False, 
                                              p=adaptiveSampleMetric/(adaptiveSampleMetric.sum()+1e-20))
        
        # # ============if have manually annotated lesion labels for training set, select half lesion-present and half lesion-absent:
        # num_lesionPatches, num_nonLesionPatches = np.sum(np.array(tumorIndicatorAll)>0.5), np.sum(np.array(tumorIndicatorAll)<=0.5)
        # assert (num_nonLesionPatches + num_lesionPatches) == len(patchIdxAll)
        # num_lesionPatches_select = min(math.ceil(self.opts.SamplePatchNumPerImage/2), num_lesionPatches)
        # num_nonLesionPatches_select = self.opts.SamplePatchNumPerImage-num_lesionPatches_select
        # if num_lesionPatches_select>=1 and num_nonLesionPatches_select>=1 and num_nonLesionPatches_select<=num_nonLesionPatches: # half lesion-present and half lesion-absent
        #     adaptiveSampleMetric = np.where(np.array(tumorIndicatorAll)>0.5, 1.0, 0.0)
        #     patchSelection_1 = np.random.choice(np.arange(len(patchIdxAll)), size=num_lesionPatches_select, replace=False, p=adaptiveSampleMetric/(adaptiveSampleMetric.sum()+1e-20))
        #     adaptiveSampleMetric = np.where(np.array(tumorIndicatorAll)<=0.5, 1.0, 0.0)
        #     patchSelection_2 = np.random.choice(np.arange(len(patchIdxAll)), size=num_nonLesionPatches_select, replace=False, p=adaptiveSampleMetric/(adaptiveSampleMetric.sum()+1e-20))
        #     patchSelection = np.concatenate([patchSelection_1, patchSelection_2], axis=0)
        # else:
        #     patchSelection = np.random.choice(np.arange(len(patchIdxAll)), size=self.opts.SamplePatchNumPerImage, replace=False, p=np.ones(len(patchIdxAll))/(len(patchIdxAll)))
        
        # =========================extract patches for current training batch=================================================
        high_pool, low_pool, lossWeight_pool = patchExtract(gt=gt, im=im, patchIdxAll=patchIdxAll, patchSelection=patchSelection, opts=self.opts, segMask=segMask, im_path=im_path)
        vol_LD, vol_HD = torch.from_numpy(np.stack(low_pool)), torch.from_numpy(np.stack(high_pool))
        vol_weight = torch.from_numpy(np.stack(lossWeight_pool)) # lesion probability

        return {'vol_low': vol_LD, 'vol_high': vol_HD, 'vol_weight': vol_weight, 'vol_path': gt_path}

    def __len__(self):
        return len(self.opts.pathAll)
