import csv
from math import cos, sin
import numpy as np
import os
import pickle

from PIL import Image
import torch
import torchvision.transforms as transforms

from data.base_dataset import BaseDataset, get_posenet_transform

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
        self.root = opt.dataroot
        driving_data = os.path.join(self.root, 'driving_data.csv')

        # select the folder corresponding to the right input type
        file_path = None
        if self.opt.input_type == 'rgb':
            file_path = os.path.join('CameraRGB0', 'image_%s.png')
        elif self.opt.input_type == 'depth':
            file_path = os.path.join('CameraDepth0', 'image_%s.png')
        else: # point clouds
            file_path = os.path.join('PointCloudLocal1', 'point_cloud_%s.pk')

        self.A_paths = []
        self.A_poses = []

        # extract driving data
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

        A = self.extract_file(A_path)

        return {'A': A, 'B': A_pose, 'A_paths': A_path}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'CloudNetDataset'

    def extract_file(self, path):
        '''
        extract the file in location [path] and transform it to make it Pytorch-compatible
        '''
        if self.opt.input_type == 'rgb':
            img = Image.open(path).convert('RGB')
            return get_posenet_transform(self.opt, np.zeros((256,256,3)))(img)

        if self.opt.input_type == 'point_cloud':
            with open(path, 'rb') as f:
                data = pickle.load(f)
            img = data[:224*224,:3].T.reshape((3,224,224))
            return torch.from_numpy(img)

        raise NotImplementedError('CloudNet does not accept this type of data for now: %s' % opt.input_type)

# def get_cloudnet_transform(opt, mean_image):
#     transform_list = []
# #    transform_list.append(transforms.Resize(opt.loadSize, Image.BICUBIC))
#     # transform_list.append(transforms.Lambda(
#     #     lambda img: __subtract_mean(img, mean_image)))
#     # transform_list.append(transforms.Lambda(
#     #     lambda img: __crop_image(img, opt.fineSize, opt.isTrain)))
#     transform_list.append(transforms.Lambda(
#         lambda img: torch.from_numpy(img)))
#     return transforms.Compose(transform_list)
