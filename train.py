import os
import numpy as np
from numpy.random import seed
import random
import time

import torch.backends.cudnn

from options.train_options import TrainOptions
from data.data_loader import create_data_loader
from models.models import create_model
from util.sql import Database
from util.util import manual_seed
from util.visualizer import Visualizer

opt = TrainOptions().parse()

## SEEDING
manual_seed(opt)

seed(opt.seed)
random.seed(opt.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## LOADING DATA
data_loader = create_data_loader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

if opt.continue_train:
    if opt.which_epoch == 'latest':
        with Database(opt.db_dir) as db:
            opt.epoch_count = db.get_last_epoch(opt.ID)+1
        opt.which_epoch = opt.epoch_count-1
    else:
        opt.epoch_count = opt.which_epoch+1

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0
lr = opt.lr

model.netG.train()

print('\n----------------- Training ------------------')

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
    loss = []
    for i, batch in enumerate(dataset):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(batch)
        model.optimize_parameters()
        #
        # if total_steps % opt.display_freq == 0:
        #     save_result = total_steps % opt.update_html_freq == 0
        #     visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)
#        if total_steps % opt.save_latest_freq == 0:
#            print('saving the latest model (epoch %d, total_steps %d)' %
#                  (epoch, total_steps))
#            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save(epoch)

        with Database(opt.db_dir) as db:
            db.update_last_epoch(opt.ID, epoch)
            db.commit()

    print('End of epoch %d / %d \t Time Taken: %d sec \t Learning rate: %s' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time, str(lr)))

    if opt.lr_policy is not None:
        tmp = model.update_learning_rate(np.median(mean_errs[:(epoch-opt.epoch_count+1)]))
        if tmp < lr:
            # TODO: sql update lr
            lr = tmp
            sql.update_lr(lr, opt.name)
