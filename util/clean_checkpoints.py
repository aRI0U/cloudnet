import os
import re
from shutil import rmtree
from sql import Database
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_DIR = os.path.join(BASE_DIR, '../checkpoints')
DB_DIR = CHECKPOINT_DIR

start = time.time()

with Database(DB_DIR) as db:
    # delete empty directories
    request = "SELECT name FROM options JOIN last_epochs ON options.id = last_epochs.id AND epoch < 100"
    print(request)
    print('Deleting empty directories...')
    for row, in db.c.execute(request):
        name = row
        print(' > Deleting %s' % name)
        try:
            rmtree(os.path.join(CHECKPOINT_DIR, name))
        except FileNotFoundError:
            pass
    db._exec("""
        DELETE FROM options
        WHERE id IN (
            SELECT id FROM last_epochs
            WHERE epoch = 0
        )
    """)
    db._exec("""
        DELETE FROM test
        WHERE id IN (
            SELECT id FROM last_epochs
            WHERE epoch = 0
        )
    """)
    db._exec("DELETE FROM last_epochs WHERE epoch < 100")
    db.commit()
    raise IndexError
    i = 0
    print('Deleting temporary models...', end='\r')
    for name, epoch in c.execute("""
        SELECT test.name, epoch FROM test
        JOIN options ON test.name = options.name
        WHERE epoch < last_epoch AND epoch % 50 != 0
    """):
        path = os.path.join(CHECKPOINT_DIR, name, '%s_net_G.pth' % epoch)
        if os.path.isfile(path):
            print('Deleting temporary models... (%d successfully deleted)' % i, end='\r')
            i += 1
            os.remove(path)
    print('%d temporary files deleted.                            ' % i)
    db.commit()

end = time.time()
print('Cleaning completed. Time elapsed: %.3fs' % (end-start))
