from numpy import load, loadtxt
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from data.base_dataset import get_posenet_transform
from util.geometry import euler_to_quaternion

class CloudNetDataset(Dataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        driving_data = os.path.join(self.root, 'poses.txt')
        # select the folder corresponding to the right input type
        img_path = os.path.join(self.root, 'CameraRGB1', 'image_%05d.png')
        pc_path = os.path.join(self.root, 'PointCloudLocal1', 'point_cloud_%05d.npy')
        self.mean_image = load(os.path.join(self.root , 'mean_image.npy'))
        frames = loadtxt(driving_data, dtype=int, usecols=0, delimiter=';', skiprows=1)
        poses = loadtxt(driving_data, dtype=float, usecols=(1,2,3,4,5,6,7), delimiter=';', skiprows=1)

        # splitting between training and test sets
        if opt.split > 0:
            if opt.isTrain or opt.phase == 'retrain':
                set = frames % opt.split != 0

            elif opt.phase == 'val':
                set = frames % opt.split == 0

            else:
                set = np.ones(len(frames), dtype=bool)

        frames = frames[set]
        self.img_paths = [img_path % f for f in frames]
        self.pc_paths = [pc_path % f for f in frames]

        self.A_poses = poses[set]
        self.transform = get_posenet_transform(opt, self.mean_image)

        self.A_size = len(self.img_paths)

    def __getitem__(self, index):
        index_A = index % self.A_size
        img_path = self.img_paths[index_A]
        pc_path = self.pc_paths[index_A]
        A_pose = self.A_poses[index_A]
        A_img = Image.open(img_path).convert('RGB')
        A_img = self.transform(A_img)
        A_pc = self._extract_file(pc_path)

        return {'X_img': A_img, 'X_pc': A_pc, 'Y': A_pose, 'X_path_img': img_path, 'X_path_pc': pc_path}

    def __len__(self):
        return self.A_size

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
            return load(path, mmap_mode='r')[:n_points, :input_nc]
        if sampling == 'uni':
            data = load(path, mmap_mode='r')
            return data[::(len(data))//n_points][:n_points, :input_nc]
        raise ValueError('Sampling [%s] does not exist' % sampling)
