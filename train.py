import numpy
import os
import random
import time

import torch

from options.train_options import TrainOptions
from data.data_loader import create_data_loader
from models.models import create_model
import util.sql as sql
from util.visualizer import Visualizer

# open connection with database
sql.connect("./checkpoints")

opt = TrainOptions().parse()

## SEEDING
torch.manual_seed(opt.seed)
numpy.random.seed(opt.seed)
random.seed(opt.seed)
# torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

## LOADING DATA
data_loader = create_data_loader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

if opt.continue_train and opt.which_epoch == 'latest':
    opt.epoch_count = sql.find_info(opt.name, 'last_epoch')[0]+1
    opt.which_epoch = str(opt.epoch_count-1)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

print('\n----------------- Training ------------------')

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
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
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)
        # if total_steps % opt.save_latest_freq == 0:
        #     print('saving the latest model (epoch %d, total_steps %d)' %
        #           (epoch, total_steps))
        #     model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save(epoch)
        sql.update_last_epoch(epoch, opt.name)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()

sql.close()
