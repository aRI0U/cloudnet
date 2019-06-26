from collections import OrderedDict
from math import pi
import os

import torch
import torch.nn.functional as F

from util.geometry import GeometricLoss, qlog

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
            self.netG = CloudCNN(opt.input_nc, opt.output_nc, opt.n_points)
        else:
            raise ValueError('Model [%s] does not exist' % model)

        if use_gpu:
            self.netG.cuda(self.gpu_ids[0])

        if (not self.isTrain or opt.continue_train) and int(opt.which_epoch) > 0:
            self.load_network(self.netG, 'G', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.mse = torch.nn.MSELoss()
            self.criterion = GeometricLoss() if self.opt.criterion == 'geo' else self.mse

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
        if self.opt.criterion == 'log':
            self.loss_ori = self.mse(qlog(self.pred_Y[:,3:]), qlog(self.input_Y[:,3:]))
        else:
            ori_gt = F.normalize(self.input_Y[:,3:], p=2, dim=1)
            self.loss_ori = self.mse(self.pred_Y[:,3:], ori_gt) * 180 / pi

        if self.opt.criterion == 'geo':
            self.loss_G = self.criterion(self.input_X[...,:3].transpose(1,2).contiguous(), self.input_Y, self.pred_Y)
        else:
            self.loss_G = self.loss_pos + self.opt.beta * self.loss_ori
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_G.step()

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
                                ('geom_err', self.loss_G.item()),
                                ])

        pos_err = torch.dist(self.pred_Y[:,:3], self.input_Y[:,:3])
        ori_gt = F.normalize(self.input_Y[:,3:], p=2, dim=1)
        abs_distance = torch.abs((ori_gt.mul(self.pred_Y[:,3:])).sum())
        ori_err = 2*180/pi * torch.acos(abs_distance)
        return [pos_err.item(), ori_err.item()]

    def get_current_pose(self):
        return self.pred_Y.data[0].cpu().numpy()

    def get_current_visuals(self):
        input_X = util.tensor2im(self.input_X.data)
        # pred_Y = util.tensor2im(self.pred_Y.data)
        # input_Y = util.tensor2im(self.input_Y.data)
        return OrderedDict([('input_X', input_X)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
