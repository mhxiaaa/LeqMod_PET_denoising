import numpy as np
from scipy.ndimage import gaussian_filter

def get_patchList_Test(img, patch_size, patch_moving_stride, validValueThresh=0.15):
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
    patchIdx = []
    for k in krange:
        for j in jrange:
            for i in irange:
                box = [k, k+patch_size[0], i, i+patch_size[1], j, j+patch_size[2]]
                imC = img[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
                if imC.max() > validValueThresh:
                    patchIdx.append(box)
    return patchIdx


def reverseImg(img_size, predictions, patchIdx):
    denoised_img = np.float32(np.zeros(img_size))
    
    patch_size = predictions[0].shape
    gaussian_mask = np.zeros(patch_size, dtype=float)
    gaussian_mask[int(patch_size[0] / 2 - 1):int(patch_size[0] / 2 + 1),
                  int(patch_size[1] / 2 - 1):int(patch_size[1] / 2 + 1),
                  int(patch_size[2] / 2 - 1):int(patch_size[2] / 2 + 1)] = 1
    gaussian_mask = gaussian_filter(gaussian_mask, patch_size[0] / 4, truncate=2, mode='nearest')
    gaussian_mask = gaussian_mask / np.amax(gaussian_mask)
    
    blending_mask = np.zeros(img_size, dtype=float)
    for ppIdx in range(len(patchIdx)):
        box = patchIdx[ppIdx]
        patchC = predictions[ppIdx]
        blending_mask[box[0]:box[1], box[2]:box[3], box[4]:box[5]] += gaussian_mask
        denoised_img[box[0]:box[1], box[2]:box[3], box[4]:box[5]] += patchC * gaussian_mask
    blending_mask = 1 / blending_mask
    denoised_img = denoised_img * blending_mask
    denoised_img[np.where(np.isnan(denoised_img))] = 0
    denoised_img[np.where(denoised_img <= 0)] = 0
    
    return denoised_img

def crop_image_zeroOut3D(img, tol=0):
    s0 = np.max(img, (1,2)) > tol
    s1 = np.max(img, (0,2)) > tol
    s2 = np.max(img, (0,1)) > tol
    x,y,z = img.shape
    s0_start, s0_end = s0.argmax(), x-s0[::-1].argmax()
    s1_start, s1_end = s1.argmax(), y-s1[::-1].argmax()
    s2_start, s2_end = s2.argmax(), z-s2[::-1].argmax()
    box = [s0_start, s0_end, s1_start, s1_end, s2_start, s2_end]
    imCrop = img[s0_start:s0_end, s1_start:s1_end, s2_start:s2_end]
    return imCrop, box