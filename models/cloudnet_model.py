from collections import OrderedDict
from math import pi
import os
import pickle

import torch
import torch.nn.functional as F

from models.mdn import MDN, mdn_loss
import util.util as util
from util.geometry import GeometricLoss

class CloudNetModel():
    def name(self):
        return 'CloudNetModel'

    def initialize(self, opt):
        print(self.name())
        self.opt = opt
        self.opt.mdn = False
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

        self.mdn = MDN(512, opt.output_nc, 3)

        if use_gpu:
            self.netG.cuda(self.gpu_ids[0])
            self.mdn.cuda(self.gpu_ids[0])

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterions = [torch.nn.MSELoss(), mdn_loss]

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
        self.image_paths = batch['X_path']
        self.input_X.resize_(input_X.size()).copy_(input_X)
        self.input_Y.resize_(input_Y.size()).copy_(input_Y)

    def forward(self):
        if self.opt.mdn:
            feature_vector = self.netG(self.input_X)
            self.pred_Y = self.mdn(feature_vector)
        else:
            self.pred_Y = self.netG(self.input_X)

    def backward(self):
        # print(self.pred_Y)
        if self.opt.mdn:
            pi, sigma, mu = self.pred_Y
            criterion = self.criterions[1]
            self.loss_pos = criterion(pi, sigma, mu[...,:3], self.input_Y[:,:3])
            self.loss_ori = criterion(pi, sigma, mu[...,3:], self.input_Y[:,3:])
            # self.regularizer = 0.1*torch.mean(sigma[...,:3])
            # pose = self.get_best_pose()
            # print('%.3f\t%.3f\t%.3f\t%.3f' % (torch.min(sigma[...,:3]).item(), torch.max(sigma[...,:3]).item(), torch.mean(sigma[...,:3]).item(), torch.median(sigma[...,:3]).item()))
            print('%.3f\t%.3f\t%.3f\t%.3f' % (torch.min(sigma[...,:3]).item(), torch.max(sigma[...,:3]).item(), torch.mean(sigma[...,:3]).item(), torch.median(sigma[...,:3]).item()))

        else:
            criterion = self.criterions[0]
            self.loss_pos = criterion(self.pred_Y[:,:3], self.input_Y[:,:3])
            self.loss_ori = criterion(self.pred_Y[:,3:], self.input_Y[:,3:])

        self.loss = (1-self.opt.beta)*self.loss_pos + self.opt.beta*self.loss_ori# + self.regularizer
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()

    # no backprop gradients
    def test(self):
        self.forward()

    def get_best_pose(self):
        if not self.opt.mdn:
            return self.pred_Y
        pi, sigma, mu = self.pred_Y
        return mu[:,torch.max(pi, dim=1).indices].squeeze(1) if pi is not None else mu.squeeze(1)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_errors(self):
        if self.opt.isTrain:
            return OrderedDict([('pos_err', self.loss_pos.item()),
                                ('ori_err', self.loss_ori.item()),
                                ('geom_err', self.loss.item()),
                                ])
        pred = self.get_best_pose()
        # print(self.pred_Y[0])
        # print(self.pred_Y[-1])
        # print([torch.dist(p[:3], self.input_Y[:,:3]).item() for p in self.pred_Y[-1].squeeze(0)])
        # pos_err = torch.dist(pred[:,:3], self.input_Y[:,:3])
        pos_err = torch.dist(pred[:,:3], self.input_Y[:,:3])
        # print(pos_err)
        ori_gt = F.normalize(self.input_Y[:,3:], p=2, dim=1)
        ori_pred = F.normalize(pred[:,3:], p=2, dim=1)
        abs_distance = torch.abs((ori_gt.mul(ori_pred)).sum())
        ori_err = 2*180/pi * torch.acos(abs_distance)
        return [pos_err.item(), ori_err.item()]

    def get_current_pose(self):
        return self.get_best_pose().data[0].cpu().numpy()



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
            'MDN_SD': self.mdn.state_dict() if self.opt.mdn else None,
            'loss': self.loss
        }, path)


    def load(self, epoch):
        filename = '%d_net_G.tar' % epoch
        path = os.path.join(self.save_dir, filename)

        checkpoint = torch.load(path)
        assert checkpoint['epoch'] == epoch
        self.netG.load_state_dict(checkpoint['network_SD'])

        try:
            mdn_dict = checkpoint['MDN_SD']
            if mdn_dict is not None:
                self.mdn.load_state_dict(mdn_dict)
        except KeyError:
            print('No state dict for MDN')
            pass

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
