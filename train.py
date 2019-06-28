import os
import numpy as np
from numpy.random import seed
import random
import time

import torch.backends.cudnn
from torch import manual_seed

from options.train_options import TrainOptions
from data.data_loader import create_data_loader
from models.models import create_model
import util.sql as sql
from util.visualizer import Visualizer

# open connection with database
sql.connect("./checkpoints")

opt = TrainOptions().parse()

## SEEDING
manual_seed(opt.seed)
seed(opt.seed)
random.seed(opt.seed)
torch.backends.cudnn.deterministic = True

## LOADING DATA
data_loader = create_data_loader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

if opt.continue_train:
    if opt.which_epoch == 'latest':
        opt.epoch_count = sql.find_info(opt.name, 'last_epoch')[0]+1
        opt.which_epoch = str(opt.epoch_count-1)
    else:
        opt.epoch_count = int(opt.which_epoch)+1

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0
lr = opt.lr

print('\n----------------- Training ------------------')
mean_errs = np.zeros(30)
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
    mean_err = 0
    for i, batch in enumerate(dataset):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(batch)
        model.optimize_parameters()

        # if total_steps % opt.display_freq == 0:
        #     save_result = total_steps % opt.update_html_freq == 0
        #     visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            mean_err += errors['geom_err']
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)
        # if total_steps % opt.save_latest_freq == 0:
        #     print('saving the latest model (epoch %d, total_steps %d)' %
        #           (epoch, total_steps))
        #     model.save('latest')
    mean_errs[(epoch-opt.epoch_count)%30] = mean_err

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save(epoch)
        sql.update_last_epoch(epoch, opt.name)

    print('End of epoch %d / %d \t Time Taken: %d sec \t Learning rate: %s' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time, str(lr)))

    if opt.lr_policy is not None:
        tmp = model.update_learning_rate(np.median(mean_errs[:(epoch-opt.epoch_count+1)]))
        if tmp < lr:
            lr = tmp
            sql.update_lr(lr, opt.name)

sql.close()
