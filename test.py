import time
import numpy as np
import os
import re

from options.test_options import TestOptions
from data.data_loader import create_data_loader
from models.models import create_model
import util.sql as sql
from util.sql import Database
from util.visualizer import Visualizer
from util import my_html

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = create_data_loader(opt)
dataset = data_loader.load_data()

checkpoints_dir = os.path.join(opt.checkpoints_dir, opt.name)
results_dir = os.path.join(opt.results_dir, opt.name)
os.makedirs(os.path.join(results_dir, 'tmp'), exist_ok=True)

besterror  = [0, float('inf'), float('inf')] # nepoch, medX, medQ

testfile = open(os.path.join(results_dir, '%s_median.txt' % opt.phase), 'a')
testfile.write('epoch medX  medQ\n')
testfile.write('==================\n')

model = create_model(opt)
visualizer = Visualizer(opt)

with Database(opt.db_dir) as db:
    epoch = 0
    last_epoch = db.get_last_epoch(opt.ID)
    while epoch < last_epoch or epoch < db.get_last_epoch(opt.ID):
        if epoch >= last_epoch:
            last_epoch = db.get_last_epoch(opt.ID)
        epoch += 1
        if not os.path.isfile(os.path.join(checkpoints_dir, '%d_net_G.tar' % epoch)):
            continue
        test_pkey = (opt.ID, epoch, opt.phase)
        if db.is_test(*test_pkey):
            continue
        db.new_test(*test_pkey)
        db.commit()

        try:
            print("epoch: %d" % epoch)
            model.load(epoch)
            model.netG.eval()
            visualizer.change_log_path(epoch)
            err = []

            for i, data in enumerate(dataset):
                model.set_input(data)
                model.test(epoch)
                img_path = model.get_image_paths()[0]
                print('\t%04d/%04d: process image... %s' % (i, len(dataset), img_path), end='\r')
                image_path = img_path.split('/')[-2] + '/' + img_path.split('/')[-1]
                pose = model.get_current_pose()
                visualizer.save_estimated_pose(image_path, pose)
                err_p, err_o = model.get_current_errors()
                # err_pos.append(err_p)
                # err_ori.append(err_o)
                err.append([err_p, err_o])

            median_pos = np.median(err, axis=0)

            if median_pos[0] < besterror[1]:
                besterror = [epoch, median_pos[0], median_pos[1]]
            print()
            # print("median position: {0:.2f}".format(np.median(err_pos)))
            # print("median orientat: {0:.2f}".format(np.median(err_ori)))
            print("\tmedian wrt pos.: {0:.2f}m {1:.2f}°".format(median_pos[0], median_pos[1]))
            testfile.write("{0:<5} {1:.2f}m {2:.2f}°\n".format(epoch,
                                                             median_pos[0],
                                                             median_pos[1]))
            testfile.flush()

            os.rename(visualizer.log_name, os.path.join(results_dir, '%s_%d.txt' % (opt.phase, epoch)))
            db.add_test_result(*test_pkey, *median_pos)

        finally:
            db.remove_tmp_test(*test_pkey)
            db.commit()

try:
    os.rmdir(os.path.join(results_dir, 'tmp'))
except OSError:
    pass

print("{0:<5} {1:.2f}m {2:.2f}°\n".format(*besterror))
testfile.write('-----------------\n')
testfile.write("{0:<5} {1:.2f}m {2:.2f}°\n".format(*besterror))
testfile.write('==================\n')
testfile.close()
