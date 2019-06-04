from collections import OrderedDict
import numpy as np
import os
import pickle

import torch
import torch.nn.functional as F

from . import networks
from .base_model import BaseModel
from util.geometry import geometric_loss

class CloudNetModel(BaseModel):
    def name(self):
        return 'CloudNetModel'

    def initialize(self, opt):
        print(self.name())
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain

        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        # define tensors
        self.input_X = self.Tensor(opt.batchSize, opt.n_points, opt.input_nc)
        self.input_Y = self.Tensor(opt.batchSize, opt.output_nc)

        # load pretrained model
        self.pretrained_weights = None
        if self.isTrain and opt.init_weights != '':
            pretrained_path = os.path.join('pretrained_models', opt.init_weights)
            print('Initializing the weights from %s...' % pretrained_path, end='\t')
            with open(pretrained_path, 'rb') as f:
                self.pretrained_weights = pickle.load(f, encoding='bytes')
            print('Done')

        # define network
        self.netG = networks.define_network(
            opt.input_nc,
            opt.n_points,
            opt.model,
            init_from=self.pretrained_weights,
            isTest=not self.isTrain,
            f_size=opt.fineSize,
            gpu_ids=self.gpu_ids
        )

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.mse = torch.nn.MSELoss()
            self.criterion = geometric_loss if self.opt.criterion == 'geo' else self.mse

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, eps=1,
                                                weight_decay=0.0625,
                                                betas=(self.opt.adambeta1, self.opt.adambeta2))
            self.optimizers.append(self.optimizer_G)
            # for optimizer in self.optimizers:
            #     self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')

    def set_input(self, batch):
        input_X = batch['X']
        input_Y = batch['Y']
        self.image_paths = batch['X_paths']
        self.input_X.resize_(input_X.size()).copy_(input_X)
        self.input_Y.resize_(input_Y.size()).copy_(input_Y)

    def forward(self):
        self.pred_Y = self.netG(self.input_X)

    def backward(self):
        # position loss
        self.loss_pos = self.mse(self.pred_Y[:,:3], self.input_Y[:,:3])
        # orientation loss
        ori_gt = F.normalize(self.input_Y[:,3:], p=2, dim=1)
        self.loss_ori = self.mse(self.pred_Y[:,3:], ori_gt) * 180 / np.pi

        if self.opt.criterion == 'mse':
            self.loss_G = self.loss_pos + self.opt.beta * self.loss_ori
        elif self.opt.criterion == 'geo':
            self.loss_G = self.criterion(self.input_X[...,:3].transpose(1,2).contiguous(), self.input_Y, self.pred_Y)
        else:
            raise AttributeError('Criterion [%s] does not exist' % self.opt.criterion)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_G.step()

    def get_current_errors(self):
        if self.opt.isTrain:
            return OrderedDict([('pos_err', self.loss_pos.item()),
                                ('ori_err', self.loss_ori.item()),
                                ('geom_err', self.loss_G.item()),
                                ])

        pos_err = torch.dist(self.pred_Y[:,:3], self.input_Y[:,:3])
        ori_gt = F.normalize(self.input_Y[:,3:], p=2, dim=1)
        abs_distance = torch.abs((ori_gt.mul(self.pred_Y[:,3:])).sum())
        ori_err = 2*180/np.pi * torch.acos(abs_distance)
        return [pos_err.item(), ori_err.item()]


    def get_current_pose(self):
        return self.pred_Y.data[0].cpu().numpy()
