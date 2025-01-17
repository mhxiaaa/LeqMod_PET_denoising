import os
import argparse
import numpy as np
import torch.utils.data
import torch.backends.cudnn as cudnn
from model_LeqModGan import modelGAN
import scipy.io as sio
import nibabel as nib
from utils_patchProcess import get_patchList_Test, reverseImg, crop_image_zeroOut3D
import matplotlib.pyplot as plt
import json
from pathlib import Path
import os
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
parser = argparse.ArgumentParser(description='PETdenoise')
parser.add_argument('--weightPath', type=str, default='....pt')
parser.add_argument('--patchBatchSize', type=int, default=40)
parser.add_argument('--validValueThresh', type=float, default=0.1, help='suv valid value')
parser.add_argument('--num_workers', type=int, default=4, help='number of threads to load data')
parser.add_argument('--patch_size', nargs='+', type=int, default=[80, 80, 80])
parser.add_argument('--stride_size', nargs='+', type=int, default=[10, 10, 10])
opts = parser.parse_args()

strSaveDir = '/LeqModGAN/'
opts.pathAll = []
# =========denoise dataset 路径===============================================================================
for file in Path('/test/').rglob('*.nii.gz'):
    opts.pathAll.append(str(file))
opts.pathAll = sorted(opts.pathAll)
print('num of images: ', len(opts.pathAll))
# ==================================================================================================================
cudnn.benchmark = True
opts.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opts.numGPUs = torch.cuda.device_count()
print('using', opts.device)
print('num of available GPUs', opts.numGPUs)

model = modelGAN(opts)
ep0, total_iter = model.resume(checkpoint_file=opts.weightPath, train=False)
model.eval()

with torch.no_grad():
    for imPath in opts.pathAll:
        savePath = imPath.replace('/test/', strSaveDir)
        if not os.path.exists(savePath):
            print('processing: ', imPath)
            imOri = nib.load(imPath)
            affine = imOri.affine
            imOri = imOri.get_fdata()

            imSUV = imOri.copy()
            imSUV[np.where(imSUV<=0)] = 0

            imCrop, boxIm = crop_image_zeroOut3D(imSUV, tol=opts.validValueThresh)
            print('min: ', imCrop.min(), '; max: ', imCrop.max())
            patchIdxAll = get_patchList_Test(img=imCrop, patch_size=opts.patch_size, patch_moving_stride=opts.stride_size, validValueThresh=opts.validValueThresh)
            # =====================================prediction========================================================
            patchAll = []
            for ppIdx in range(0, len(patchIdxAll), opts.patchBatchSize):
                if ppIdx <= (len(patchIdxAll)-opts.patchBatchSize):
                    BatchSizeCurrent = opts.patchBatchSize
                else:
                    BatchSizeCurrent = len(patchIdxAll) - (len(patchIdxAll)//opts.patchBatchSize) * opts.patchBatchSize
                
                patchesCurrent = []
                for bbIdx in range(BatchSizeCurrent):
                    boxPP = patchIdxAll[ppIdx+bbIdx]
                    lowS = imCrop[boxPP[0]:boxPP[1], boxPP[2]:boxPP[3], boxPP[4]:boxPP[5]]
                    patchesCurrent.append(lowS)
                
                lowS = torch.from_numpy(np.array(patchesCurrent)).to(opts.device).float().unsqueeze(dim=1)
                predS = model.net_G(lowS.detach())
                predS[predS<=0.0] = 0.0
                
                predS = predS * lowS.mean(dim=(1,2,3,4)).unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).unsqueeze(dim=4) \
                                / predS.mean(dim=(1,2,3,4)).unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).unsqueeze(dim=4)
                predS = predS[:,0,:,:,:].detach().cpu().numpy()
                predS[np.where(np.isnan(predS))] = 0
                predS[np.where(predS <= 0)] = 0

                for bbIdx in range(BatchSizeCurrent):
                    patchAll.append(np.squeeze(predS[bbIdx,:,:,:]))
            
            predIm = reverseImg(img_size=imCrop.shape, predictions=patchAll, patchIdx=patchIdxAll)
            
            # ========================================save pred results=================================================
            predFinal = np.zeros_like(imSUV)
            predFinal[boxIm[0]:boxIm[1], boxIm[2]:boxIm[3], boxIm[4]:boxIm[5]] = predIm
            
            
            os.makedirs(os.path.dirname(savePath), exist_ok=True)
            nib.save(nib.Nifti1Image(predFinal, affine=affine), savePath)
            # nib.save(nib.Nifti1Image(predFinal, affine=np.eye(4)), savePath)
            print('saving: ', savePath)