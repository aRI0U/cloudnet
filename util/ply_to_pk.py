import numpy as np
import os
import pickle

SOURCE_PATH = '/storage/group/hodl4cv/lopc/driving_project'
DEST_PATH = '../datasets/Carla'

def convert(source, dest):
    if os.path.isfile(dest):
        return 0
    print(' > Converting %s...' % source, end='\t')
    data = np.loadtxt(source, skiprows=10, delimiter=' ')
    with open(dest, 'wb') as f:
        pickle.dump(data, f)
    print('Done')

def explore(source, dest):
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
            explore(dir, os.path.join(dest, file))
        elif len(file) > 4 and file[-4:] == '.ply':
            os.makedirs(dest, exist_ok=True)
            new_file = '%s.pk' % file[:-4]
            convert(os.path.join(source, file), os.path.join(dest, new_file))

if __name__ == '__main__':
    explore(SOURCE_PATH, DEST_PATH)
