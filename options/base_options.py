import argparse
from datetime import datetime
import os
import pickle

import torch
from util.sql import Database
# import util.sql as sql

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False
        self.base = self.parser.add_argument_group('Base options')
        self.param = self.parser.add_argument_group('Experiment parameters')

    def initialize(self):
        # self.required.add_argument('--dataroot', required=True, help='path to images')
        self.base.add_argument('--dataroot', type=str, default='./datasets/Carla/episode_000', help='path to point clouds')
        self.base.add_argument('--adambeta1', type=float, default=0.9, help='first momentum term of adam')
        self.base.add_argument('--adambeta2', type=float, default=0.999, help='second momentum term of adam')
        self.base.add_argument('--batchSize', type=int, default=32, help='input batch size')
        self.base.add_argument('--beta', type=float, default=0.8, help='beta factor used in posenet.')
        self.base.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.base.add_argument('--criterion', type=str, choices=['geo','log','mse','multi'], default='mse', help='criterion used as loss')
        self.base.add_argument('--dataset_mode', type=str, choices=['unaligned','aligned','single'], default='unaligned', help='the way datasets are loaded')
        self.param.add_argument('--db_dir', type=str, default=None, help='location of the database containing options infos')
        self.param.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.param.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.param.add_argument('--display_winsize', type=int, default=224,  help='display window size')
        self.base.add_argument('--fineSize', type=int, default=224, help='then crop to this size')
        self.param.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.base.add_argument('--input_nc', type=int, default=6, help='# of input image channels')
        self.base.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        self.base.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        self.base.add_argument('--lr_policy', type=str, default=None, help='learning rate policy: lambda|step|plateau')
        self.base.add_argument('--lr_decay_iters', type=int, default=20, help='multiply by a gamma every lr_decay_iters iterations')
        self.base.add_argument('--lstm_hidden_size', type=int, default=256, help='hidden size of the LSTM layer in PoseLSTM')
        self.base.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.base.add_argument('--model', type=str, choices=['posenet','poselstm','cloudnet','cloudnetvlad'], default='cloudcnn', help='model to use')
        self.param.add_argument('--name', type=str, default=None, help='name of the experiment. It decides where to store samples and models')
        self.base.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.base.add_argument('--no_flip', action='store_true', default=True, help='if specified, do not flip the images for data augmentation')
        self.param.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
        self.base.add_argument('--n_points', default=16384, type=int, help='# points considered per point cloud')
        self.base.add_argument('--output_nc', type=int, default=7, help='# of output image channels')
        self.base.add_argument('--resize_or_crop', type=str, default='scale_width_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.base.add_argument('--sampling', type=str, choices=['fps', 'uni'], default='fps', help='chooses whether points are sampled uniformly or by farthest point sampling')
        self.param.add_argument('--seed', type=int, default=0, help='initial random seed for deterministic results')
        self.base.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.base.add_argument('--split', type=int, default=0, help='split dataset into training and validation set')

        self.initialized = True

    def get_group_attr(self, group):
        return {a.dest: getattr(self.opt,a.dest,None) for a in group._group_actions}

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test
        self.opt.ID = None

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        if self.opt.db_dir is None:
            self.opt.db_dir = self.opt.checkpoints_dir

        with Database(self.opt.db_dir) as db:
            db.init_tables()
            id = None
            if self.opt.name is None:
                if not self.isTrain or self.opt.continue_train:
                    # find experiment with options
                    self.opt.ID = db.find_experiment(**self.get_group_attr(self.base))
                if self.opt.ID is None:
                    if self.isTrain:
                        # create new experiment
                        self.opt.name = os.path.join(self.opt.model, datetime.now().strftime('%Y-%m-%d_%H:%M'))
                        self.opt.continue_train = False
                    else:
                        raise ValueError("There is no model trained with such options")

            else:
                if not self.isTrain or self.opt.continue_train:
                    self.opt.ID = db.find_experiment(name=self.opt.name)
                    print(self.opt.ID)
                    print('Loading options from the experiment...', end='\t')
                    opt_vals, opt_names = db.find_info(self.opt.ID, '*', get_col_names=True)

                    for i, opt in enumerate(opt_names):
                        if opt == 'id':
                            continue
                        opt_type = eval('type(self.opt.%s)' % opt)
                        if opt_vals[i] is None:
                            fmt = '%s'
                        elif opt_type == bool:
                            fmt = 'bool(%s)'
                        elif opt_type == int:
                            fmt = '%s'
                        elif opt_type == float:
                            fmt = 'float("%s")'
                        elif opt_type == str:
                            fmt = '"%s"'
                        else:
                            raise TypeError("Unparsed type: %s" % str(type))
                        exec(('self.opt.%s = '+fmt) % (opt, opt_vals[i]))
                    print('Done.')
                if self.opt.ID is None:
                    if self.isTrain:
                        self.opt.continue_train = False
                    else:
                        raise ValueError("There is no model with such a name: %s" % self.opt.name)

            if self.opt.beta > 1:
                self.opt.beta = self.opt.beta/(self.opt.beta+1)

            args = vars(self.opt)

            # save to database
            if self.opt.isTrain and not self.opt.continue_train:
                self.opt.ID = db.new_experiment(**self.get_group_attr(self.base), name=self.opt.name)
                db.commit()

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, 'opt_'+self.opt.phase+'.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        return self.opt
