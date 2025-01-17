import monai
from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
from nets_UNet_Seg import Unet, gaussian_weights_init
from torch.nn import MSELoss, L1Loss
from torch.optim import lr_scheduler
import torch.nn.functional as F

class unet(nn.Module):
    def __init__(self, opts):
        super(unet, self).__init__()
        self.optimizers = []
        self.loss_names = []
        self.is_train = True if hasattr(opts, 'lr') else False
        self.opts = opts

        self.net_seg = Unet(n_channels=1, n_classes=2, c=[32, 64, 128, 256])
        print('Num of total paras in net_seg: {}'.format(sum([p.numel() for p in self.net_seg.parameters()])))
        
        self.net_seg.to(opts.device)
        if opts.numGPUs >= 1:
            self.net_seg = nn.DataParallel(self.net_seg, device_ids=range(opts.numGPUs))
            
        if self.is_train:
            self.optimizer_seg = torch.optim.Adam(self.net_seg.parameters(), lr=opts.lr, betas=(0.9, 0.999), weight_decay=1e-7)
            self.optimizers.append(self.optimizer_seg)
            
            self.loss_names += ['loss_segSupervision']

    def initialize(self):
        if self.opts.preWeight is not None:
            self.net_seg.module.load_state_dict(torch.load(self.opts.preWeight)['net_seg'], strict=True)
            print('networks intialized from: ', self.opts.preWeight)
        else:
            self.net_seg.apply(gaussian_weights_init)
            print('networks intialized from Gaussian_weights')

    def set_scheduler(self, opts, epoch=-1):
        if opts.lr_policy == 'ReduceLROnPlateau':
            self.schedulers = [lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opts.gamma, 
                               patience=opts.Plateau_step_size, cooldown=1, verbose=True) for optimizer in self.optimizers]
        else:
            NotImplementedError('learning rate policy [%s] is not implemented', opts.lr_policy)

    def set_input(self, data):
        self.vol_img = data['vol_img'].flatten(0,1).to(self.opts.device).float().unsqueeze(dim=1)  # [B,1,X,Y,Z]
        self.vol_img[self.vol_img <= 0.0] = 0.0
        self.vol_gt = data['vol_gt'].flatten(0,1).to(self.opts.device).float().unsqueeze(dim=1)
        self.vol_gt[self.vol_gt <= 0.0] = 0.0

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def optimize(self, total_iter): 
        # plt.imsave('/home4/mx79/c1.png', self.vol_img[10,0,:,:,20].detach().cpu().numpy(), cmap='jet')
        # plt.imsave('/home4/mx79/c2.png', self.vol_gt[10,0,:,:,20].detach().cpu().numpy(), cmap='jet')
        self.net_seg.zero_grad()

        seg0, seg1, seg2, seg3 = self.net_seg(self.vol_img)
        cW = torch.tensor([0.1, 0.9]).to(self.opts.device)
        maskOneHot = torch.cat((1.0-self.vol_gt.long(), self.vol_gt.long()), dim=1)
        self.loss_segSupervision = self.opts.weightLoss_segSupervision * (
                                    1.0 * nn.CrossEntropyLoss(weight=cW)(seg0, self.vol_gt[:,0,:,:,:].long()) + \
                                    0.5 * monai.losses.DiceFocalLoss(softmax=True, squared_pred=True, reduction='mean')(seg0, maskOneHot) + \
                                    0.5 * nn.CrossEntropyLoss(weight=cW)(seg1, F.interpolate(self.vol_gt, seg1.shape[2::])[:,0,:,:,:].long()) + \
                                    0.1 * nn.CrossEntropyLoss(weight=cW)(seg2, F.interpolate(self.vol_gt, seg2.shape[2::])[:,0,:,:,:].long()) + \
                                    0.05 * nn.CrossEntropyLoss(weight=cW)(seg3, F.interpolate(self.vol_gt, seg3.shape[2::])[:,0,:,:,:].long()))
        
        self.loss_segSupervision.backward()
        self.optimizer_seg.step()

    @property
    def loss_summary(self):
        message = 'segSupervision: {:4f}'.format(self.loss_segSupervision.item())
        return message

    def update_learning_rate(self, epochLossAll):
        for scheduler in self.schedulers:
            if self.opts.lr_policy == 'ReduceLROnPlateau':
                scheduler.step(epochLossAll)
        
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = {:7f}'.format(lr))
        return lr

    def save(self, filename, epoch, total_iter):
        state = {}
        state['net_seg'] = self.net_seg.module.state_dict()
        state['opt_seg'] = self.optimizer_seg.state_dict()
        state['epoch'] = epoch
        state['total_iter'] = total_iter
        torch.save(state, filename)
        print('Saved {}'.format(filename))

    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file)
        self.net_seg.module.load_state_dict(checkpoint['net_seg'])
        if train:
            self.optimizer_seg.load_state_dict(checkpoint['opt_seg'])
        print('Loaded {}'.format(checkpoint_file))
        return checkpoint['epoch'], checkpoint['total_iter']
