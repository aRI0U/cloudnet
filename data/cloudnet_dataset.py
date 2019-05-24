import csv
import numpy as np
import os
import pickle

from PIL import Image
import torch
import torchvision.transforms as transforms

from data.base_dataset import BaseDataset, get_posenet_transform
from util.geometry import euler_to_quaternion

class CloudNetDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        driving_data = os.path.join(self.root, 'driving_data.csv')

        # select the folder corresponding to the right input type
        file_path = None
        if self.opt.model in ['posenet', 'poselstm']:
            file_path = os.path.join('CameraRGB0', 'image_%s.png')
        elif self.opt.model in ['cloudnet', 'cloudcnn']:
            file_path = os.path.join('PointCloudLocal1', 'point_cloud_%s.pickle')
        else:
            raise AttributeError('Model [%s] does not exist' % self.opt.model)
        self.mean_image = None #np.load(os.path.join(self.root , 'mean_image.npy'))
        self.A_paths = []
        self.A_poses = []

        # extract driving data
        print('Extracting data from %s...' % driving_data, end='\t')
        with open(driving_data, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_number = row['name'][-5:]
                path = os.path.join(self.root, file_path % img_number)
                # ignore the line if the point cloud doesn't exist
                if not os.path.isfile(path):
                    continue
                self.A_paths.append(path)
                pose = np.array([row['pos_x'], row['pos_y'], row['pos_z']], dtype=float)
                # for now only 1 dof of rotation is enabled
                orientation = np.array([0,0,row['steer']], dtype=float)/2
                pose = np.concatenate((pose, euler_to_quaternion(*orientation)))
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
        if self.opt.model in ['posenet', 'poselstm']:
            img = Image.open(path).convert('RGB')
            return get_posenet_transform(self.opt, self.mean_image)(img)

        if self.opt.model in ['cloudnet', 'cloudcnn']:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            img = self._sample(data, self.opt.input_nc, self.opt.n_points, self.opt.sampling)
            return torch.from_numpy(img)
        raise ValueError('Model [%s] does not exist' % self.opt.model)

    @staticmethod
    def _sample(data, input_nc, n_points, sampling):
        if sampling == 'fps':
            return data[:n_points, :input_nc]
        if sampling == 'uni':
            return data[::(len(data))//n_points][:n_points, :input_nc]
        raise ValueError('Sampling [%s] does not exist' % sampling)
