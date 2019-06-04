import numpy as np
import os
import sqlite3

relevant_options = ['batchSize', 'beta', 'criterion', 'dataset_mode','input_nc','max_dataset_size','model','no_dropout','n_points', 'output_nc', 'sampling', 'seed', 'serial_batches']

connection = None

def connect(db_dir):
    global connection
    connection = sqlite3.connect(os.path.join(db_dir, "options.db"))

def close():
    connection.close()

def new_experiment(args):
    c = connection.cursor()
    command = "INSERT INTO options VALUES ("
    for _, v in sorted(args.items()):
        command += "'%s'," % str(v)
    command += "0)"
    try:
        c.execute(command)
    except sqlite3.OperationalError:
        c.execute("DROP TABLE IF EXISTS options")
        create_db = "CREATE TABLE options ("
        for k, _ in sorted(args.items()):
            create_db += "%s," % str(k)
        create_db += "last_epoch)"
        c.execute(create_db)
        c.execute(command)
    connection.commit()

def add_experiment(dir, last_epoch):
    opt_train = os.path.join(dir, 'opt_train.txt')
    with open(opt_train, 'r') as f:
        command = "INSERT INTO options ("
        values = " VALUES ("
        for row in f.readlines():
            data = row.split(': ')
            if len(data) == 2:
                command += "%s," % data[0]
                values += "'%s'," % data[1][:-1]
        command += 'last_epoch)'
        values += '%d)' % last_epoch
    c = connection.cursor()
    c.execute(command + values)
    connection.commit()

def find_experiment(opt):
    command = "SELECT MAX(name) FROM options WHERE "
    for o in relevant_options:
        command += "%s == '%s'" % (o, eval('str(opt.%s)' % o))
        command += " AND "
    command += "beta == '%s'" % opt.beta
    c = connection.cursor()
    c.execute(command)
    name = c.fetchone()[0]
    return name

def find_info(name, cols):
    c = connection.cursor()
    c.execute("SELECT %s FROM options WHERE name = '%s'" % (cols, name))
    return c.fetchone()

def update_last_epoch(epoch, name):
    c = connection.cursor()
    c.execute("UPDATE options SET last_epoch = ? WHERE name = ?", (epoch, name))
    connection.commit()

def init_test_db():
    c = connection.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS test (name, epoch, pos_err, ori_err)")

def add_test_result(name, epoch, pos_err, ori_err):
    c = connection.cursor()
    c.execute("INSERT INTO test (name, epoch, pos_err, ori_err) VALUES (?,?,?,?)", (name, epoch, pos_err, ori_err))
    connection.commit()

def get_test_result(name, epoch=None):
    c = connection.cursor()
    if epoch is None:
        c.execute("SELECT epoch, pos_err, ori_err FROM test WHERE name = ?", (name,))
        return c.fetchall()
    c.execute("SELECT pos_err, ori_err FROM test WHERE name = ? AND epoch = ?", (name, epoch))
    return c.fetchone()


if __name__ == '__main__':
    # add_experiment('checkpoints/cloudcnn/2019-05-31_16:18', 50)
    # find_experiment()
    connect("./checkpoints")
    c = connection.cursor()
    # c.execute('DELETE FROM options WHERE name = ""')
    # c.execute("UPDATE options SET last_epoch = 130 WHERE name = 'cloudcnn/2019-06-01_11:13'")
    # for row in c.execute('SELECT n_points, sampling, criterion, name, last_epoch FROM options ORDER BY n_points'):
    #     print(row)
    for row in c.execute("SELECT * FROM TEST"):
        print(row)
    # connection.commit()
    close()
