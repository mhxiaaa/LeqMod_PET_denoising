import os
import argparse
import json
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from utils_dataloader3D import TrainSet
from model_unet_OnlySeg import unet
import csv
import numpy as np
import random
import pickle
from pathlib import Path
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1,0"
parser = argparse.ArgumentParser(description='PETjoint')
parser.add_argument('--output_path', type=str, default='/00.expRun_seg1/')
parser.add_argument('--experiment_name', type=str, default='unet_seg')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--preWeight', type=str, default=None)
parser.add_argument('--num_workers', type=int, default=4, help='number of threads to load data')
parser.add_argument('--weightLoss_segSupervision', default=2.0)
parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
parser.add_argument('--SamplePatchNumPerImage', type=int, default=35, help='select partial patches')
parser.add_argument('--patch_size', nargs='+', type=int, default=[80, 80, 80], help='patch size for train')
parser.add_argument('--stride_size', nargs='+', type=int, default=[40, 40, 40], help='patch size for train')
parser.add_argument('--validValueThresh', type=float, default=0.2, help='suv valid value for filter patches')
parser.add_argument('--AUG', default=True, action='store_true', help='use augmentation')
parser.add_argument('--rotate_train', type=int, default=10, help='randomly rotate patch along z for train')
parser.add_argument('--saveModel_epochs', type=int, default=1, help='save model for every number of epochs')
parser.add_argument('--saveLoss_epochs', type=int, default=1, help='save loss info for every number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=500, help='number of epoch')
parser.add_argument('--lr_policy', type=str, default='ReduceLROnPlateau', help='ReduceLROnPlateau/multistep/cosine/')
parser.add_argument('--gamma', type=float, default=0.1, help='decay ratio for step scheduler')
parser.add_argument('--Plateau_step_size', type=int, default=6, help='step size for scheduler')
opts = parser.parse_args()

opts.imgDirSeg = '/00._datasetSeg/'
opts.boundBoxDirSeg = '/00._datasetSeg_boundBox/'
opts.pathAll = []  # all paths to imgs
for file in Path('/00._datasetSeg/').rglob('*SUV.nii.gz'):
    opts.pathAll.append(str(file))
print('total training pairs:', len(opts.pathAll))
# ====================================================================================================================
output_directory = os.path.join(opts.output_path, opts.experiment_name)
checkpoint_directory = os.path.join(output_directory, 'checkpoints')
if not os.path.exists(checkpoint_directory):
    print("Creating directory: {}".format(checkpoint_directory))
    os.makedirs(checkpoint_directory)
with open(os.path.join(output_directory, 'options.json'), 'w') as f:
    f.write(json.dumps(opts.__dict__, indent=4, sort_keys=False))
# ==================================================================================================================
cudnn.benchmark = True
opts.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opts.numGPUs = torch.cuda.device_count()
print('using', opts.device)
print('num of available GPUs', opts.numGPUs)

model = unet(opts)
if opts.resume is None:
    model.initialize()
    ep0, total_iter = -1, 0
else:
    ep0, total_iter = model.resume(opts.resume)

model.set_scheduler(opts, ep0)
ep0 += 1
print('Start training at epoch {} \n'.format(ep0))

train_set = TrainSet(opts)
train_loader = DataLoader(dataset=train_set, num_workers=opts.num_workers, batch_size=opts.batch_size, shuffle=True, pin_memory=True)

with open(os.path.join(output_directory, 'train_loss.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(model.loss_names)
# ======================================training loop===============================================
for epoch in range(ep0, opts.n_epochs + 1):
    epochLoss = []

    train_bar = tqdm(train_loader)
    model.train()
    model.set_epoch(epoch)
    for it, data in enumerate(train_bar):
        total_iter += 1
        model.set_input(data)
        model.optimize(total_iter)
        train_bar.set_description(desc='[Epoch {}]'.format(epoch) + model.loss_summary)
        epochLoss.append(list(model.get_current_losses().values()))
    
    epochLossAll = np.mean(np.array(epochLoss), axis=0)
    current_lr = model.update_learning_rate(epochLossAll.mean())
    
    if (epoch+1) % opts.saveLoss_epochs == 0: # write loss info
        with open(os.path.join(output_directory, 'train_loss.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(np.mean(np.array(epochLoss), axis=0))

    if (epoch+1) % opts.saveModel_epochs == 0: # save checkpoint
        print('Saving checkpoint ......')
        checkpoint_name = os.path.join(checkpoint_directory, 'model_{}.pt'.format(epoch))
        model.save(checkpoint_name, epoch, total_iter)
    
    if current_lr < 1e-7:
        print('Terminating training. Learning rate below threshold.')
        break
