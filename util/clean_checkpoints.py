import os
import re
from shutil import rmtree
import sqlite3
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_DIR = os.path.join(BASE_DIR, '../checkpoints')
DB_DIR = CHECKPOINT_DIR

start = time.time()
connection = sqlite3.connect(os.path.join(DB_DIR, "options.db"))
c = connection.cursor()

# delete empty directories
print('Deleting empty directories...')
for row in c.execute("SELECT name FROM options WHERE last_epoch = 0"):
    name = row[0]
    print(' > Deleting %s' % name)
    try:
        rmtree(os.path.join(CHECKPOINT_DIR, name))
    except FileNotFoundError:
        pass
c.execute("DELETE FROM options WHERE last_epoch = 0")

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
connection.commit()
connection.close()
end = time.time()
print('Cleaning completed. Time elapsed: %.3fs' % (end-start))
