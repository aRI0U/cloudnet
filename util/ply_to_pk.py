import numpy as np
import os
import glob
import random
import sys
import time
from tqdm import tqdm

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH = '/storage/group/hodl4cv/lopc/driving_project'
DEST_PATH = '../datasets/Carla'

def _distances(p0, pts, d=3):
    return torch.norm((p0.unsqueeze(0)-pts)[:,:d], dim=1)

def fps(data, npoint, first=None):
    r"""
        Farthest point sampling (greedy algorithm)

        Parameters
        ----------
        data: torch.FloatTensor
            (N,d) tensor containing point cloud
        npoint: int
            number of points that have to be selected
        first: int (optional)
            index of first point selected (random if first is None)

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
    farthest_pts[0] = data[random.randint(0,N-1)] if first is None else data[first]
    distances = _distances(farthest_pts[0], data, d=min(d,3))
    for i in range(1,npoint):
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
    print(' > Converting %s...' % source)
    pc = np.loadtxt(source, skiprows=10, delimiter=' ', dtype=float)
    data = torch.from_numpy(pc)
    start = time.time()
    xyz = fps(data, len(data))
    xyz_numpy = xyz.cpu().numpy()
    with open(dest, 'wb') as f:
        pickle.dump(xyz_numpy, f)
    end = time.time()
    print('Done. Time elpased: %.3f' % (end-start))

if __name__ == '__main__':
    print('Conversion of local point clouds from %s...' % SOURCE_PATH)
    os.chdir(SOURCE_PATH)
    for file in glob.iglob(os.path.join('./**/*.ply'), recursive=True):
        new_file = os.path.join(DEST_PATH, file[:-3] + 'npy')
        convert(file, new_file, False)
