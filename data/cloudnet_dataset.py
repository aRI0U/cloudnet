import csv
from numpy import load
import os

import torch
from torch.utils.data import Dataset

from util.geometry import euler_to_quaternion

class CloudNetDataset(Dataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        driving_data = os.path.join(self.root, 'driving_data.csv')
        # select the folder corresponding to the right input type
        file_path = None
        file_path = os.path.join('PointCloudLocal1', 'point_cloud_%s.npy')
        self.mean_image = None #np.load(os.path.join(self.root , 'mean_image.npy'))
        self.A_paths = []
        self.A_poses = []

        # extract driving data
        print('Extracting data from %s...' % driving_data, end='\t')
        with open(driving_data, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_number = row['name'][-5:]
                if opt.split > 0:
                    if (opt.isTrain or opt.phase == 'retrain') and int(img_number) % opt.split == 0:
                        continue
                    if opt.phase == 'val' and int(img_number) % opt.split != 0:
                        continue
                path = os.path.join(self.root, file_path % img_number)
                # ignore the line if the point cloud doesn't exist
                if not os.path.isfile(path):
                    continue
                self.A_paths.append(path)
                pose = torch.tensor((float(row['pos_x']), float(row['pos_y']), float(row['pos_z'])))
                # for now only 1 dof of rotation is enabled
                pose = torch.cat((pose, euler_to_quaternion(0, 0, float(row['steer'])/2)))
                self.A_poses.append(pose)
        print('Done')
        self.A_size = len(self.A_paths)

    def __getitem__(self, index):
        index_A = index % self.A_size
        A_path = self.A_paths[index % self.A_size]
        A_pose = self.A_poses[index % self.A_size]

        A = self._extract_file(A_path)

        return {'X': A, 'Y': A_pose, 'X_paths': A_path}

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
