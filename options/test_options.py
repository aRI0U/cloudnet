from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self):
        BaseOptions.__init__(self)
        self.test = self.parser.add_argument_group('Test options')

    def initialize(self):
        BaseOptions.initialize(self)
        self.test.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.test.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.test.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.test.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        # self.test.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')

        self.isTrain = False
