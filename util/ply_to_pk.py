import numpy as np
import os
import pickle
import random
import sys
import time
from tqdm import tqdm

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH = '/storage/group/hodl4cv/lopc/driving_project/episode_000'
DEST_PATH = '../datasets/Carla/episode_000'

N_POINTS = 2**16

def _distances(p0, pts, d=3):
    return torch.norm((p0.unsqueeze(0)-pts)[:,:d], dim=1)

def fps(data, npoint):
    r"""
        Farthest point sampling (greedy algorithm)

        Parameters
        ----------
        data: torch.FloatTensor
            (N,d) tensor containing point cloud
        npoint: int
            number of points that have to be selected

        Returns
        -------
        torch.cuda.FloatTensor
            (npoint,d) tensor containing the selected points
    """
    data = data.cuda()
    N, d = data.shape
    if N < npoint:
        return data
    farthest_pts = torch.randn(npoint, d, dtype=torch.double).cuda()
    farthest_pts[0] = data[random.randint(0,N-1)]
    distances = _distances(farthest_pts[0], data, d=min(d,3))
    for i in tqdm(range(1,npoint)):
        pt = data[distances.argmax()]
        distances = torch.min(distances, _distances(pt, data, d=min(d,3)))
        farthest_pts[i] = pt
    return farthest_pts

def convert(source, dest, replace):
    r"""
        Point clouds converter

        Parameters
        ----------
        source: str
            location of the point cloud
        dest: str
            location where the new point cloud is supposed to be written
    """
    if not replace and os.path.isfile(dest):
        return 0
    print(' > Converting %s...' % source, end='\t')
    pc = np.loadtxt(source, skiprows=10, delimiter=' ', dtype=float)
    data = torch.from_numpy(pc)
    start = time.time()
    xyz = fps(data, len(data))
    xyz_numpy = xyz.cpu().numpy()
    with open(dest, 'wb') as f:
        pickle.dump(xyz_numpy, f)
    end = time.time()
    print('Done. Time elpased: %.3f' % (end-start))

def explore(source, dest, replace):
    '''
    Explores recursively [source] and converts every .ply file in it to a binary
    file. Binary files are stocked in [dest], with the same architecture as in
    [source]
    '''
    print('Exploring %s...' % source)
    for file in os.listdir(source):
        dir = os.path.join(source, file)
        if os.path.isdir(dir):
            if len(file) == 11 and 'PointCloud' in file: # do not convert global point clouds
                continue
            explore(dir, os.path.join(dest, file), replace)
        elif len(file) > 5 and file[-5:] == '%s.ply' % sys.argv[1]: # partitionning the dataset to launch script on multiple machines
            os.makedirs(dest, exist_ok=True)
            new_file = '%s.pickle' % file[:-4]
            convert(os.path.join(source, file), os.path.join(dest, new_file), replace)

if __name__ == '__main__':
    replace = False
    explore(SOURCE_PATH, os.path.join(BASE_DIR, DEST_PATH), replace)
    print('Done')
