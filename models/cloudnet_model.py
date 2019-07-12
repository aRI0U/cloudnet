from collections import OrderedDict
from math import pi
import os
import pickle

import torch
import torch.nn.functional as F

import util.util as util
from util.geometry import GeometricLoss

class CloudNetModel():
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

        # define network
        use_gpu = len(self.gpu_ids) > 0

        if use_gpu:
            assert(torch.cuda.is_available())

        if opt.model == 'cloudnet':
            from models.cloudnet import CloudNet
            self.netG = CloudNet(opt.input_nc, opt.output_nc, opt.n_points)
        elif opt.model == 'cloudcnn':
            from models.cloudcnn import CloudCNN
            self.netG = CloudCNN(opt.input_nc, opt.output_nc, opt.n_points, use_gpu=use_gpu)
        else:
            raise ValueError('Model [%s] does not exist' % model)

        if use_gpu:
            self.netG.cuda(self.gpu_ids[0])


        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterion = None
            if self.opt.criterion == 'geo':
                self.criterion = GeometricLoss()
            elif self.opt.criterion == 'multi':
                self.criterion = torch.nn.MSELoss(reduction='none')
            else:
                self.criterion = torch.nn.MSELoss()

            self.optimizer = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, eps=1,
                                                weight_decay=0.001,
                                                betas=(self.opt.adambeta1, self.opt.adambeta2))

            self.scheduler = None

            # if opt.lr_policy == 'plateau':
            #     self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #         self.optimizer,
            #         mode='min',
            #         factor=0.1,
            #         patience=opt.lr_decay_iters,
            #         cooldown=0,
            #         verbose=True
            #     )

            if opt.continue_train and int(opt.which_epoch) > 0:
                self.load(opt.which_epoch)

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
        pred_pos, input_pos = self.pred_Y[...,:3], self.input_Y[:,:3].unsqueeze(1)
        pred_ori, input_ori = self.pred_Y[...,3:self.opt.output_nc], self.input_Y[:,3:self.opt.output_nc].unsqueeze(1)
        sigma = torch.exp(self.pred_Y[...,-2])
        alpha = F.softmax(self.pred_Y[...,-1])
        self.loss_pos = torch.sum(torch.log(torch.sum(alpha/(sigma**3) * torch.exp(-F.mse_loss(pred_pos - input_pos, dim=-1)/(2*sigma**2)), dim=-1)))
        self.loss_ori = torch.sum(torch.log(torch.sum(alpha/(sigma**3) * torch.exp(-F.mse_loss(pred_ori - input_ori, dim=-1)/(2*sigma**2)), dim=-1)))
        self.loss = (1-self.opt.beta)*self.loss_pos + self.opt.beta*self.loss_ori

        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()

    # no backprop gradients
    def test(self):
        self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_errors(self):
        if self.opt.isTrain:
            return OrderedDict([('pos_err', self.loss_pos.item()),
                                ('ori_err', self.loss_ori.item()),
                                ('geom_err', self.loss.item()),
                                ])

        pos_err = torch.dist(self.pred_Y[:,:3], self.input_Y[:,:3])
        ori_gt = F.normalize(self.input_Y[:,3:], p=2, dim=1)
        abs_distance = torch.abs((ori_gt.mul(self.pred_Y[:,3:self.opt.output_nc])).sum())
        ori_err = 2*180/pi * torch.acos(abs_distance)
        return [pos_err.item(), ori_err.item()]

    def get_current_pose(self):
        return self.pred_Y.data[0].cpu().numpy()

    def get_current_visuals(self):
        input_X = util.tensor2im(self.input_X.data)
        # pred_Y = util.tensor2im(self.pred_Y.data)
        # input_Y = util.tensor2im(self.input_Y.data)
        return OrderedDict([('input_X', input_X)])



    def save(self, epoch):
        filename = '%d_net_G.tar' % epoch
        path = os.path.join(self.save_dir, filename)

        torch.save({
            'epoch': epoch,
            'network_SD': self.netG.state_dict(),
            'optimizer_SD': self.optimizer.state_dict(),
            'loss': self.loss
        }, path)


    def load(self, epoch):
        filename = '%s_net_G.tar' % epoch
        path = os.path.join(self.save_dir, filename)

        checkpoint = torch.load(path)
        assert checkpoint['epoch'] == int(epoch)
        self.netG.load_state_dict(checkpoint['network_SD'])

        if self.isTrain:
            self.optimizer.load_state_dict(checkpoint['optimizer_SD'])
            self.loss = checkpoint['loss']

            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_SD'])


    # update learning rate (called once every epoch)
    def update_learning_rate(self, val_loss):
        self.scheduler.step(val_loss)
        print(val_loss)
        print(self.scheduler.best)
        return self.optimizers[0].param_groups[0]['lr']
