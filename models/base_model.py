from collections import OrderedDict
import numpy as np
import os
import pickle

import torch

class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        print(self.name())
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # define tensors
        self.input_X = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_Y = self.Tensor(opt.batchSize, opt.output_nc)

        # load pretrained model
        self.pretrained_weights = None
        if self.isTrain and opt.init_weights != '':
            pretrained_path = os.path.join('pretrained_models', opt.init_weights)
            print('Initializing the weights from %s...' % pretrained_path, end='\t')
            with open(pretrained_path, 'rb') as f:
                self.pretrained_weights = pickle.load(f, encoding='bytes')
            print('Done')

    def set_input(self, batch):
        input_X = batch['X']
        input_Y = batch['Y']
        self.image_paths = batch['X_paths']
        self.input_X.resize_(input_X.size()).copy_(input_X)
        self.input_Y.resize_(input_Y.size()).copy_(input_Y)

    def forward(self):
        pass

    # no backprop gradients
    def test(self):
        self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        if self.opt.isTrain:
            return OrderedDict([('pos_err', self.loss_pos),
                                ('ori_err', self.loss_ori),
                                ])

        pos_err = torch.dist(self.pred_Y[0], self.input_Y[:, 0:3])
        ori_gt = F.normalize(self.input_Y[:, 3:], p=2, dim=1)
        abs_distance = torch.abs((ori_gt.mul(self.pred_Y[1])).sum())
        ori_err = 2*180/numpy.pi * torch.acos(abs_distance)
        return [pos_err.item(), ori_err.item()]


    def get_current_pose(self):
        return np.concatenate((self.pred_Y[0].data[0].cpu().numpy(),
                                  self.pred_Y[1].data[0].cpu().numpy()))

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
