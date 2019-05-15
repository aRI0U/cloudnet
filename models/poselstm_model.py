import numpy as np
import torch
import torch.nn.functional as F
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy

class PoseLSTModel(BaseModel):
    def name(self):
        return 'PoseLSTModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.mean_image = np.load(os.path.join(opt.dataroot , 'mean_image.npy'))

        self.netG = networks.define_network(opt.input_nc, opt.lstm_hidden_size, opt.model,
                                      init_from=self.pretrained_weights, isTest=not self.isTrain,
                                      gpu_ids = self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterion = torch.nn.MSELoss()

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
        # networks.print_network(self.netG)
        # print('-----------------------------------------------')

    def forward(self):
        self.pred_Y = self.netG(self.input_X)

    def backward(self):
        self.loss_G = 0
        self.loss_pos = 0
        self.loss_ori = 0
        loss_weights = [0.3, 0.3, 1]
        for l, w in enumerate(loss_weights):
            mse_pos = self.criterion(self.pred_Y[2*l], self.input_Y[:, 0:3])
            ori_gt = F.normalize(self.input_Y[:, 3:], p=2, dim=1)
            mse_ori = self.criterion(self.pred_Y[2*l+1], ori_gt)
            self.loss_G += (mse_pos + mse_ori * self.opt.beta) * w
            self.loss_pos += mse_pos.item() * w
            self.loss_ori += mse_ori.item() * w * self.opt.beta
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_G.step()
