from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        BaseOptions.__init__(self)
        self.train = self.parser.add_argument_group('Train options')

    def initialize(self):
        BaseOptions.initialize(self)
        self.train.add_argument('--display_freq', type=int, default=64, help='frequency of showing training results on screen')
        self.train.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.train.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        self.train.add_argument('--print_freq', type=int, default=64, help='frequency of showing training results on console')
        self.train.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.train.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        self.train.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.train.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.train.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.train.add_argument('--which_epoch', type=int, default=None, help='which epoch to load? set to None to use latest cached model')
        self.train.add_argument('--niter', type=int, default=500, help='# of iter at starting learning rate')
        self.train.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        self.train.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.train.add_argument('--use_html', action='store_true', help='save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.train.add_argument('--init_weights', type=str, default='places-googlenet.pickle', help='initiliaze network from, e.g., places-googlenet.pickle')

        self.isTrain = True
