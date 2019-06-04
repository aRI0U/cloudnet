import matplotlib.pyplot as plt
import numpy as np
import sys

import util.sql as sql

XLABEL = sys.argv[1] if len(sys.argv) > 1 else 'epoch'

class Options():
    def __init__(self, options, n_points):
        self.batchSize = 32
        self.beta = 5
        self.dataset_mode = 'unaligned'
        self.input_nc = 6
        self.max_dataset_size = float('inf')
        self.no_dropout = False
        self.output_nc = 7
        self.seed = 0
        self.serial_batches = False


        self.checkpoints_dir = './checkpoints'
        self.results_dir = './results'
        self.model = 'cloudcnn'
        self.n_points = n_points
        self.name = options
        splitted = options.split('-')
        self.criterion = splitted[0]
        self.sampling = splitted[1]

OPTIONS = ['mse-fps', 'geo-fps', 'mse-uni']

sql.connect('./checkpoints')

data = np.zeros((3, 10, 30, 2))

for o in range(len(OPTIONS)):
    for i in range(6, 16):
        opt = Options(OPTIONS[o], 2**i)
        name = sql.find_experiment(opt)
        if name is None:
            continue
        errors = np.array(sql.get_test_result(name))
        if len(errors) == 0:
            continue
        epochs = (errors[:,0]//10).astype(int)
        data[o,i-6,epochs] = errors[:,1:]

sql.close()

data = 1/(1+data)
data[:,:,0] = 0

if XLABEL == 'n_points':
    epoch = 50
    for o in range(len(OPTIONS)):
        plt.plot(np.arange(6,16), data[o,:,epoch//10,0], label=OPTIONS[o])
    plt.legend()
    plt.title('Accuracy as a function of point cloud size for epoch ')
    plt.xlabel('log(n_points)')
    plt.ylabel('Accuracy')
    plt.show()

elif XLABEL == 'epoch':
    n_points = 6
    for o in range(len(OPTIONS)):
        plt.plot(np.arange(0, 300, 10), data[o,n_points-6,:,0], label=OPTIONS[o])
    plt.legend()
    plt.title('Accuracy as a function of the number of epochs (n_points = %d)' % (2**n_points))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

elif XLABEL == 'both':
    option = 0
    for i in range(6, 16):
        plt.plot(np.arange(0, 300, 10), data[option,i-6,:,0], label=str(2**i))
    plt.legend()
    plt.title('Accuracy as a function of the number of epochs for option %s' % OPTIONS[option])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

else:
    raise ValueError('Accuracy can be plotted as a function of {epoch, n_points, both}. Please choose between these options.')
