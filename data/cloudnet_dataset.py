import numpy as np
import os

import torch
from torch.utils.data import Dataset

class CloudNetDataset(Dataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        driving_data = os.path.join(self.root, 'poses.txt')
        # select the folder corresponding to the right input type
        pc_path = os.path.join(self.root, 'PointCloudLocal1', 'point_cloud_%05d.npy')
        self.mean_image = np.load(os.path.join(self.root , 'mean_image.npy'))
        frames = np.loadtxt(driving_data, dtype=int, usecols=0, delimiter=';', skiprows=1)
        poses = np.loadtxt(driving_data, dtype=float, usecols=(1,2,3,4,5,6,7), delimiter=';', skiprows=1)

        # splitting between training and test sets
        set = np.ones(len(frames), dtype=bool)
        if opt.split > 0:
            if opt.isTrain or opt.phase == 'retrain':
                set = frames % opt.split != 0

            elif opt.phase == 'val':
                set = frames % opt.split == 0

        frames = frames[set]
        self.pc_paths = [pc_path % f for f in frames]

        self.poses = poses[set]

        self.size = len(self.pc_paths)

    def __getitem__(self, index):
        index_A = index % self.size
        pc_path = self.pc_paths[index_A]
        pose = self.poses[index_A]

        pc = self._extract_file(pc_path)

        return {'X': pc, 'Y': pose, 'X_path': pc_path}

    def __len__(self):
        return self.size

    def name(self):
        return 'CloudNetDataset'

    def _extract_file(self, path):
        # type: (CloudNetDataset, str) -> torch.FloatTensor
        r"""
            Extracts a file and transform it into a usable tensor

            Parameters
            ----------
            path: str
                location of the file that has to be transformed

            Returns
            -------
            torch.FloatTensor
                (opt.n_points,opt.input_nc) tensor of point cloud
        """
        img = self._sample(path, self.opt.input_nc, self.opt.n_points, self.opt.sampling)
        return torch.from_numpy(img)

    @staticmethod
    def _sample(path, input_nc, n_points, sampling):
        if sampling == 'fps':
            return np.load(path, mmap_mode='r')[:n_points, :input_nc]
        if sampling == 'uni':
            data = np.load(path, mmap_mode='r')
            return data[::(len(data))//n_points][:n_points, :input_nc]
        raise ValueError('Sampling [%s] does not exist' % sampling)
