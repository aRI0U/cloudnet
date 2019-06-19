import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
import util.sql as sql

XLABEL = sys.argv[1] if len(sys.argv) > 1 else 'epoch'
EPOCH_MAX = int(sys.argv[3]) if len(sys.argv) > 3 else float('inf')
DIM = 0

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

def accuracy(loss):
    return 1/(1+(loss/10))

OPTIONS = ['mse-fps', 'geo-fps', 'mse-uni', 'log-fps']

sql.connect('../checkpoints')

if XLABEL == 'n_points':
    epoch = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    for o in range(len(OPTIONS)):
        plt.plot(np.arange(6,16), data[o,:,epoch//10,0], label=OPTIONS[o])
    plt.legend()
    plt.title('Accuracy as a function of point cloud size for epoch ')
    plt.xlabel('log(n_points)')
    plt.ylabel('Accuracy')
    plt.show()

elif XLABEL == 'epoch':
    n_points = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
    xmax = 0
    for o in OPTIONS:
        name = sql.find_experiment(Options(o, n_points))
        if name is None:
            print('No experiment with such parameters: (%s,%d)' % (o, n_points))
            continue
        err = sql.get_test_result(name)
        if err is None:
            print('No experiment with such parameters: (%s,%d)' % (o, n_points))
            continue
        err = np.array(err)
        err = err[err[:,1]!=None]
        xmax = max(xmax, max(err[:,0]))
        plt.plot(err[:,0], accuracy(err[:,DIM+1]), label=o)
    sql.close()
    xmax = min(xmax, EPOCH_MAX)
    plt.plot([0,xmax], [1/1.1,1/1.1], ':', c='b', label='medX=1')
    plt.plot([0,xmax], [0.5,0.5], ':', c='r', label='medX=10')
    print(EPOCH_MAX)
    plt.xlim(0,xmax)
    plt.ylim(0,1)
    plt.legend(loc='upper left')
    plt.title('Accuracy as a function of the number of epochs (n_points = %d)' % (n_points))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

elif XLABEL == 'both':
    option = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    for i in range(6, 16):
        plt.plot(np.arange(0, 300, 10), data[option,i-6,:,0], label=str(2**i))
    plt.legend()
    plt.title('Accuracy as a function of the number of epochs for option %s' % OPTIONS[option])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

else:
    raise ValueError('Accuracy can be plotted as a function of {epoch, n_points, both}. Please choose between these options.')
