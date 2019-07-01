import time
import numpy as np
import os
import re

from options.test_options import TestOptions
from data.data_loader import create_data_loader
from models.models import create_model
import util.sql as sql
from util.visualizer import Visualizer
from util import my_html

sql.connect("./checkpoints")

sql.init_test_db()

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = create_data_loader(opt)
dataset = data_loader.load_data()

checkpoints_dir = os.path.join(opt.checkpoints_dir, opt.name)
results_dir = os.path.join(opt.results_dir, opt.name)
os.makedirs(results_dir, exist_ok=True)

besterror  = [0, float('inf'), float('inf')] # nepoch, medX, medQ

testfile = open(os.path.join(results_dir, 'test_median.txt'), 'a')
testfile.write('epoch medX  medQ\n')
testfile.write('==================\n')

# if opt.which_epoch == 'latest':
#     last_epoch = sql.find_info(opt.name, 'last_epoch')[0]
#     opt.which_epoch = str(last_epoch)

model = create_model(opt)
visualizer = Visualizer(opt)

testepochs = []
for pth in os.listdir(checkpoints_dir):
    epoch = re.search('(.+?)_net_G.pth', pth)
    if epoch is None:
        continue
    testepochs.append(epoch.group(1))

for testepoch in sorted(testepochs, key = lambda s: '%6s' % s):
    if sql.get_test_result(opt.name, opt.phase, int(testepoch)) is not None:
        continue

    sql.new_test(opt.name, int(testepoch), opt.phase)

    try:
        model.load_network(model.netG, 'G', testepoch)
        visualizer.change_log_path(testepoch)
        # test
        # err_pos = []
        # err_ori = []
        err = []
        print("epoch: %s" % testepoch)
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.test()
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
        sql.add_test_result(opt.name, int(testepoch), opt.phase, median_pos[0], median_pos[1])

    except KeyboardInterrupt:
        sql.remove_empty_tests()
        test_file = os.path.join(results_dir, '%s_%s.txt' % (opt.phase, testepoch))
        if os.path.isfile(test_file):
            os.remove(test_file)
        raise KeyboardInterrupt

    if median_pos[0] < besterror[1]:
        besterror = [testepoch, median_pos[0], median_pos[1]]
    print()
    # print("median position: {0:.2f}".format(np.median(err_pos)))
    # print("median orientat: {0:.2f}".format(np.median(err_ori)))
    print("\tmedian wrt pos.: {0:.2f}m {1:.2f}°".format(median_pos[0], median_pos[1]))
    testfile.write("{0:<5} {1:.2f}m {2:.2f}°\n".format(testepoch,
                                                     median_pos[0],
                                                     median_pos[1]))
    testfile.flush()
print("{0:<5} {1:.2f}m {2:.2f}°\n".format(*besterror))
testfile.write('-----------------\n')
testfile.write("{0:<5} {1:.2f}m {2:.2f}°\n".format(*besterror))
testfile.write('==================\n')
testfile.close()

sql.close()
