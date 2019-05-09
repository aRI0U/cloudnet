import csv
from math import cos, sin
import numpy as np
import os

from PIL import Image
import plyfile
import torch

from data.base_dataset import BaseDataset, get_posenet_transform

DATASET_PATH = '/storage/group/hodl4cv/lopc/driving_project/episode_%03d'

def euler_to_quaternion(theta, phi, psi):
    '''
    converts an orientation represented by Euler angles into a unitary quaternion
    '''
    return np.array([
        cos(phi)*cos(theta)*cos(psi) + sin(phi)*sin(theta)*sin(psi),
        sin(phi)*cos(theta)*cos(psi) - cos(phi)*sin(theta)*sin(psi),
        cos(phi)*sin(theta)*cos(psi) + sin(phi)*cos(theta)*sin(psi),
        cos(phi)*cos(theta)*sin(psi) - sin(phi)*sin(theta)*cos(psi),
    ])


class CloudNetDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = DATASET_PATH % int(opt.dataroot)
        driving_data = os.path.join(self.root, 'driving_data.csv')

        # select the folder corresponding to the right input type
        file_path = None
        if self.opt.input_type == 'rgb':
            file_path = os.path.join('CameraRGB0', 'image_%s.png')
        elif self.opt.input_type == 'depth':
            file_path = os.path.join('CameraDepth0', 'image_%s.png')
        else: # point clouds
            file_path = os.path.join('PointCloudLocal0', 'point_cloud_%s.ply')

        # check if the input data exist
        if not os.path.isfile(os.path.join(self.root, file_path % '00000')):
            raise ValueError('No available data of type %s' % self.opt.input_type)

        self.A_paths = []
        self.A_poses = []
        # read csv file
        print('Extracting data from %s...' % driving_data, end='\t')
        with open(driving_data, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_number = row['name'][-5:]
                self.A_paths.append(os.path.join(self.root, file_path % img_number))
                pose = np.array([row['pos_x'], row['pos_y'], row['pos_z']], dtype=float)
                orientation = np.array([row['steer'],0,0], dtype=float)/2
                pose = np.concatenate((pose, euler_to_quaternion(*orientation)))
                self.A_poses.append(pose)
        self.A_poses = np.array(self.A_poses)
        print('Done')

        self.A_size = len(self.A_paths)

    def __getitem__(self, index):
        index_A = index % self.A_size
        A_path = self.A_paths[index % self.A_size]
        A_pose = self.A_poses[index % self.A_size]

        A = self.transform(A_path)

        return {'A': A, 'B': A_pose, 'A_paths': A_path}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'CloudNetDataset'

    def transform(self, path):
        if self.opt.input_type == 'rgb':
            img = Image.open(path).convert('RGB')
            return get_posenet_transform(self.opt, np.zeros((256,256,3)))(img)

        if self.opt.input_type == 'point_cloud':
            data = plyfile.PlyData.read(path)['vertex']
            xyz = np.c_[data['x'], data['y'], data['z']]
            xyz = xyz[:2]
            return torch.from_numpy(xyz.T)
        raise NotImplementedError('CloudNet does not accept this type of data for now: %s' % opt.input_type)
