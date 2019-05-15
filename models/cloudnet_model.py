from collections import OrderedDict
from math import pi

import torch
import torch.nn.functional as F

from . import networks
from .base_model import BaseModel
from util.geometry import geometric_loss

class CloudNetModel(BaseModel):
    def name(self):
        return 'CloudNetModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # self.mean_image = np.load(os.path.join(opt.dataroot , 'mean_image.npy'))

        self.netG = networks.define_network(
            opt.input_nc,
            opt.lstm_hidden_size,
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
            self.criterion = geometric_loss
            self.mse = torch.nn.MSELoss()

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
        for i, w in enumerate(loss_weights):
            # position loss
            mse_pos = self.mse(self.pred_Y[i][:,:3], self.input_Y[:,0:3])
            # orientation loss
            ori_gt = F.normalize(self.input_Y[:,3:], p=2, dim=1)
            mse_ori = self.mse(self.pred_Y[i][:,3:], ori_gt)

            self.loss_pos += mse_pos.item() * w
            self.loss_ori += mse_ori.item() * w * 180 / pi
            loss = self.criterion(self.input_X.reshape(-1,3,self.opt.fineSize**2), self.input_Y, self.pred_Y[i])
            self.loss_G += w * loss
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_G.step()

    def get_current_errors(self):
        if self.opt.isTrain:
            return OrderedDict([('pos_err', self.loss_pos),
                                ('ori_err', self.loss_ori),
                                ('geom_err', self.loss_G.item()),
                                ])

        raise NotImplementedError
        pos_err = torch.dist(self.pred_Y[0], self.input_Y[:, 0:3])
        ori_gt = F.normalize(self.input_Y[:, 3:], p=2, dim=1)
        abs_distance = torch.abs((ori_gt.mul(self.pred_Y[1])).sum())
        ori_err = 2*180/numpy.pi * torch.acos(abs_distance)
        return [pos_err.item(), ori_err.item()]
