import glob
import os
import shutil
import sys
import time

from sql import Database

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(BASE_DIR, '..')
checkpoints = os.path.join(root, 'checkpoints')

def get_exps(name):
    os.chdir(checkpoints)
    return glob.iglob(name)

print('These action will definitely delete all data about the following experiments:')
for name in sys.argv[1:]:
    for exp in get_exps(name):
        print(exp)

if input('Continue? [y/n] ').lower() in 'yes':
    start = time.time()

    for n in sys.argv[1:]:
        for name in get_exps(n):
            print('Performing deletion for %s' % name)
            # destroy results
            results = os.path.join(root, 'results')
            try:
                shutil.rmtree(os.path.join(results, name))
            except FileNotFoundError:
                pass

            try:
                shutil.rmtree(os.path.join(checkpoints, name))
            except FileNotFoundError:
                pass

            with Database(checkpoints) as db:
                for table in ['options', 'test', 'tmp']:
                    db._exec("DELETE FROM %s WHERE name = ?" % table, (name,))
                db.commit()

    end = time.time()
    print('Done. Time elapsed: %.3f' % (end-start))

else:
    print('Aborted.')
