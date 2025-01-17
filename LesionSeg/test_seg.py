import os
import argparse
import numpy as np
import torch.utils.data
import torch.backends.cudnn as cudnn
import model_unet_OnlySeg
import nibabel as nib
from utils_patchProcess_new import get_patchList_Train, reverseImg
import matplotlib.pyplot as plt
import json
from pathlib import Path
import pickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
parser = argparse.ArgumentParser(description='PETseg')
parser.add_argument('--output_path', type=str, default=' ')
parser.add_argument('--experiment_name', type=str, default='unet_seg')
parser.add_argument('--weightPath', type=str, default='checkpoints/segUnet.pt')
parser.add_argument('--patchBatchSize', type=int, default=20)
parser.add_argument('--num_workers', type=int, default=8, help='number of threads to load data')
parser.add_argument('--patch_size', nargs='+', type=int, default=[80, 80, 80], help='patch size for train')
parser.add_argument('--stride_size', nargs='+', type=int, default=[20, 20, 20], help='patch size for train')
parser.add_argument('--validValueThresh', type=float, default=0.2, help='suv valid value')
opts = parser.parse_args()

output_directory = os.path.join(opts.output_path, opts.experiment_name)
segSave_directory = os.path.join(output_directory, 'segOnDenoiseDataHC')

opts.imgDirSeg = ' /'
opts.boundBoxDirSeg = ' /'
opts.pathAll = []  # all paths to imgs
for file in Path('.../').rglob('*SUV.nii.gz'):
    opts.pathAll.append(str(file))
# ==================================================================================================================
cudnn.benchmark = True
opts.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opts.numGPUs = torch.cuda.device_count()
print('using', opts.device)
print('num of available GPUs', opts.numGPUs)

opts.resume = os.path.join(output_directory, opts.weightPath)
model = model_unet_OnlySeg.unet(opts)
ep0, total_iter = model.resume(opts.resume, train=False)

model.eval()
with torch.no_grad():
    for idx in range(0, len(opts.pathAll)):
        gtPath = opts.pathAll[idx] # imPath
        
        boundBoxPath = gtPath.replace(opts.imgDir, opts.boundBoxDir).replace('100.nii.gz', '100.json')
        with open(boundBoxPath, 'r') as f:
            boxIm = json.load(f)   
        
        # seg on high-count
        savePath_segOnHC = os.path.join(segSave_directory, gtPath.split('/')[-2]+'/' + gtPath.split('/')[-1])
        if not os.path.exists(savePath_segOnHC):
            print('processing seg on:', gtPath)
            gt = nib.load(gtPath).get_fdata()
            gtCrop = gt[boxIm[0]:boxIm[1], boxIm[2]:boxIm[3], boxIm[4]:boxIm[5]]

            patchIdxAll = get_patchList_Train(img=gtCrop, patch_size=opts.patch_size, patch_moving_stride=opts.stride_size, validValueThresh=opts.validValueThresh)
            patchAllseg = []
            for ppIdx in range(0, len(patchIdxAll), opts.patchBatchSize):
                if ppIdx <= (len(patchIdxAll)-opts.patchBatchSize):
                    BatchSizeCurrent = opts.patchBatchSize
                else:
                    BatchSizeCurrent = len(patchIdxAll) - (len(patchIdxAll)//opts.patchBatchSize) * opts.patchBatchSize
                
                patchesCurrent = []
                for bbIdx in range(BatchSizeCurrent):
                    boxPP = patchIdxAll[ppIdx+bbIdx]
                    imS = gtCrop[boxPP[0]:boxPP[1], boxPP[2]:boxPP[3], boxPP[4]:boxPP[5]]
                    patchesCurrent.append(imS)
                
                imS = torch.from_numpy(np.array(patchesCurrent)).to(opts.device).float().unsqueeze(dim=1) # [B,1,80,80,80]
                segS, _,_,_, = model.net_seg(imS.detach())
                segS = torch.softmax(segS, dim=1)[:,1,:,:,:].detach().cpu().numpy()

                for bbIdx in range(BatchSizeCurrent):
                    patchAllseg.append(np.squeeze(segS[bbIdx,:,:,:]))
            
            seg = reverseImg(img_size=gtCrop.shape, predictions=patchAllseg, patchIdx=patchIdxAll)
            # ========================================save pred results=================================================
            segFinal = np.zeros_like(gt)
            segFinal[boxIm[0]:boxIm[1], boxIm[2]:boxIm[3], boxIm[4]:boxIm[5]] = seg
            if not os.path.exists(os.path.dirname(savePath_segOnHC)):
                # print("Creating directory: {}".format(os.path.dirname(savePath_segOnHC)))
                os.makedirs(os.path.dirname(savePath_segOnHC))
            nib.save(nib.Nifti1Image(segFinal, affine=np.eye(4)), savePath_segOnHC)
            print('saving: ', savePath_segOnHC)
