import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from data.base_dataset import get_posenet_transform

def qlog(q):
    n = np.minimum(np.linalg.norm(q[:,1:], axis=1), 1e-8)
    log = q[:,:1] * np.arccos(np.clip(q[:,1:], -1, 1))
    return log/n[:,np.newaxis]


class CloudNetDataset(Dataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        driving_data = os.path.join(self.root, 'poses.txt')
        # select the folder corresponding to the right input type
        pc_path = os.path.join(self.root, 'PointCloudLocal1', 'point_cloud_%05d.npy')

        if self.opt.model == 'posepoint':
            img_path = os.path.join(self.root, 'CameraRGB1', 'image_%05d.png')
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

        if self.opt.model == 'posepoint':
            self.img_paths = [img_path % f for f in frames]
            self.transform = get_posenet_transform(opt, self.mean_image)

        self.poses = poses[set]

        if opt.criterion == 'log':
            self.poses = np.concatenate((self.poses[:,:3], qlog(self.poses[:,3:])), axis=1)

        self.size = len(self.pc_paths)

    def __getitem__(self, index):
        index_A = index % self.size
        pc_path = self.pc_paths[index_A]
        pose = self.poses[index_A]
        pc = self._extract_file(pc_path)

        if self.opt.model == 'posepoint':
            img_path = self.img_paths[index_A]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)

            return {'X_img': img, 'X_pc': pc, 'Y': pose, 'img_path': img_path, 'pc_path': pc_path}
        return {'X_pc': pc, 'Y': pose, 'pc_path': pc_path}

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
        pc = self._sample(path, self.opt.input_nc, self.opt.n_points, self.opt.sampling)
        return torch.from_numpy(pc)

    @staticmethod
    def _sample(path, input_nc, n_points, sampling):
        if sampling == 'fps':
            return np.load(path, mmap_mode='r')[:n_points, :input_nc]
        if sampling == 'uni':
            data = np.load(path, mmap_mode='r')
            return data[::(len(data))//n_points][:n_points, :input_nc]
        raise ValueError('Sampling [%s] does not exist' % sampling)
