import os
import nibabel as nib
import json
import numpy as np
from utils_patchProcess import crop_image_zeroOut3D
from pathlib import Path

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

dataDir = '' # path to the dir of images
boundBoxDir = '' # path for saving the bounding box

imgPathAll = []
for file in Path(dataDir).rglob('*_100.nii.gz'):
    imgPathAll.append(str(file))

for idx in range(len(imgPathAll)):
    imPath = imgPathAll[idx]
    print('processing: ', idx, '; imgPath: ', imPath)

    savePath = imPath.replace('100.nii.gz', '100.json').replace(dataDir, boundBoxDir)
    if not os.path.exists(os.path.dirname(savePath)):
        os.makedirs(os.path.dirname(savePath))

    im = nib.load(imPath).get_fdata()
    imCrop, box = crop_image_zeroOut3D(im, tol=0.2) # move out the dark outer bound regions in 3D images, with a thresh of SUV=0.2
    print('oriShape:', im.shape, ' cropShape', imCrop.shape)

    with open(savePath, 'w') as f:
        json.dump(box, f, indent=2, cls=NpEncoder)
    print('saving:', savePath)