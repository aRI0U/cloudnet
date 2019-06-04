import os
import re
from shutil import rmtree
import sqlite3
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_DIR = os.path.join(BASE_DIR, '../checkpoints')
DB_DIR = CHECKPOINT_DIR

def delete_tmp_models(source):
    # type: str -> unit
    r"""
        Deleting temporary saved epochs. For each training, we only keep 1/10 of
        the saved epochs (we always keep each hundred epoch)

        Parameters
        ----------
        source: str
            path of the training directory
    """
    pths = []
    epoch_max = 0
    for file in os.listdir(source):
        match = re.search('(.+?)_net_G.pth', file)
        if match is not None:
            epoch = match.group(1)
            if epoch != 'latest':
                epoch = int(epoch)
                pths.append(epoch)
                epoch_max = max(epoch_max, epoch)
    for epoch in pths:
        if epoch % (epoch_max//10) != 0 and epoch % 100 != 0:
            file_path = os.path.join(source, '%d_net_G.pth' % epoch)
            print(' > Deleting %s...' % file_path, end='\t')
            os.remove(file_path)
            print('Done')


def explore_and_clean(source):
    # type: str -> unit
    r"""
        Recursive exploration of checkpoints directory. Too short trainings and
        empty directories are deleted.

        Parameters
        ----------
        source: str
            path of the recursively explored directory
    """
    print('Exploring %s...' % source)
    is_empty = True
    for file in os.listdir(source):
        is_empty = False
        path = os.path.join(source, file)
        if os.path.isdir(path):
            explore_and_clean(path)
        elif os.path.isfile(path) and file == 'opt_train.txt':
            if len([f for f in os.listdir(source)]) < 3:
                print(' > Deleting %s...' % source, end='\t')
                rmtree(source)
                cleaned = True
                print('Done')
                break
            # delete_tmp_models(source)
    if is_empty:
        print(' > Deleting %s...' % source, end='\t')
        rmtree(source)
        print('Done')


if __name__ == '__main__':
    cleaned = True
    start = time.time()
    connection = sqlite3.connect(os.path.join(DB_DIR, "options.db"))
    c = connection.cursor()
    for row in c.execute("SELECT name FROM options WHERE last_epoch = 0"):
        name = row[0]
        print('Deleting %s' % name)
        try:
            rmtree(os.path.join(CHECKPOINT_DIR, name))
        except FileNotFoundError:
            pass
    c.execute("DELETE FROM options WHERE last_epoch <= 10")
    connection.commit()
    connection.close()
    end = time.time()
    print('Cleaning completed. Time elapsed: %.3fs' % (end-start))
