import os
import re
import sqlite3
import glob

relevant_options = ['batchSize', 'criterion', 'dataset_mode','input_nc','model','no_dropout','n_points', 'output_nc', 'sampling', 'seed', 'serial_batches', 'split']

connection = None

def connect(db_dir):
    r"""
        Open the connection with database "options.db"

        Parameters
        ----------
        db_dir: str
            location of the database
    """
    global connection
    connection = sqlite3.connect(os.path.join(db_dir, "options.db"))

def close():
    r"""
        Close the connection with database "options.db"
    """
    connection.close()


def new_experiment(args):
    r"""
        Add a new experiment to database when it begins

        Parameters
        ----------
        args: # TODO:
    """
    c = connection.cursor()
    name = args['name']
    c.execute("INSERT INTO options (name) VALUES (?)", (name,))
    for k in args:
        v = args[k]
        if type(v) == list:
            v = str(v)
        c.execute("UPDATE options SET %s = ? WHERE name = ?" % k, (v,name))
    c.execute("UPDATE options SET last_epoch = 0 WHERE name = ?", (name,))
    connection.commit()

def _add_experiment(dir):
    r"""
        Add a previous experiment that were not added to the database

        Parameters
        ----------
        dir: str
            location of the experiment
        last_epoch: int
            epoch of the latest saved model
    """
    opt_train = os.path.join(dir, 'opt_train.txt')
    last_epoch = 0
    for f in os.listdir(dir):
        if '_net_G.pth' in f:
            last_epoch = max(last_epoch, int(re.search('(.+?)_net_G.pth', f).group(1)))
    with open(opt_train, 'r') as f:
        command = "INSERT INTO options ("
        values = " VALUES ("
        for row in f.readlines():
            data = row.split(': ')
            if len(data) == 2:
                k, v = data[0], data[1][:-1]
                if v in ['True','False']:
                    v = int(bool(v))
                else:
                    try:
                        i = int(v)
                    except ValueError:
                        v = "'%s'" % v
                command += "%s," % k
                values += "%s," % v
        command += 'last_epoch)'
        values += '%d)' % last_epoch
    c = connection.cursor()
    c.execute(command + values)
    # connection.commit()

def find_experiment(opt):
    r"""
        Experiment finder

        Parameters
        ----------
        opt: options.base_options.BaseOptions
            parameters of the experiment

        Returns
        -------
        str
            name of the experiment (None if there is no such experiment)
    """
    command = "SELECT MAX(name) FROM options WHERE "
    for o in relevant_options:
        v = eval('opt.%s' % o)
        if type(v) == bool:
            v = int(bool(v))
        elif str(v) == 'inf':
            v = "'inf'"
        else:
            try:
                i = float(v)
            except ValueError:
                v = "'%s'" % v
        command += "%s = %s" % (o, v)
        command += " AND "
    command += "beta = %s" % opt.beta
    c = connection.cursor()
    c.execute(command)
    name = c.fetchone()
    if name is None:
        return None
    return name[0]

def find_info(name, cols, get_col_names=False):
    r"""
        Extract information from database

        Parameters
        ----------
        name: str
            name of the experiment
        cols: str
            cols whose data has to be extracted

        Returns
        -------
        tuple
            tuple containing data from experiment whose name is [name]
        or
        (tuple, list) tuple
            (data from experiment, colnames)
    """
    c = connection.cursor()
    c.execute("SELECT %s FROM options WHERE name = ?" % cols, (name,))
    if get_col_names:
        description = []
        for d in c.description:
            description.append(d[0])
        return c.fetchone(), description
    return c.fetchone()

def update_last_epoch(epoch, name):
    r"""
        Update epoch of latest saved model in database

        Parameters
        ----------
        epoch: int
            epoch
        name: str
            name of experiment
    """
    c = connection.cursor()
    c.execute("UPDATE options SET last_epoch = ? WHERE name = ?", (epoch, name))
    connection.commit()

def init_test_db():
    r"""
        Create (if not exists) the database containing test results
    """
    c = connection.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS test (name TEXT, epoch INTEGER, pos_err REAL, ori_err REAL, phase TEXT)")

def new_test(name, epoch, phase):
    r"""
        Initialize a new test

        Parameters
        ----------
        name: str
            name of the experiment
        epoch: int
            epoch of the model
    """
    c = connection.cursor()
    c.execute("INSERT INTO test (name, epoch, phase) VALUES (?,?,?)", (name, epoch, phase))
    connection.commit()

def add_test_result(name, epoch, phase, pos_err, ori_err):
    r"""
        Add test result in the database

        Parameters
        ----------
        name: str
            name of the experiment
        epoch: int
            epoch of the model
        pos_err: float
            error on pose
        ori_err: float
            error on orientation
    """
    c = connection.cursor()
    c.execute("UPDATE test SET pos_err = ?, ori_err = ? WHERE name = ? AND epoch = ? AND phase = ?", (pos_err, ori_err, name, epoch, phase))
    connection.commit()

def get_test_result(name, phase, epoch=None):
    r"""
        Extract test results from database

        Parameters
        ----------
        name: str
            name of the experiment
        epoch: int (optional)
            epoch of model tested

        Returns
        -------
        tuples list if epoch is None else tuple
            results of test for epoch if epoch is not None else every epoch
    """
    c = connection.cursor()
    if epoch is None:
        c.execute("SELECT epoch, pos_err, ori_err FROM test WHERE name = ? AND phase = ? ORDER BY epoch", (name,phase))
        return c.fetchall()

    c.execute("SELECT pos_err, ori_err FROM test WHERE name = ? AND phase = ? AND epoch = ?", (name, phase, epoch))
    return c.fetchone()

def _delete_test(name, epoch):
    c = connection.cursor()
    c.execute("DELETE FROM test WHERE name = ? AND epoch = ?", (name, epoch))
    connection.commit()

def remove_empty_tests():
    c = connection.cursor()
    c.execute("DELETE FROM test WHERE pos_err IS NULL OR ori_err IS NULL")
    connection.commit()

def init_db(args):
    c = connection.cursor()
    args['last_epoch'] = 0
    colnames = '(%s)' % ','.join((k for k in args))
    c.execute("CREATE TABLE IF NOT EXISTS options %s" % colnames)

def reinit_db(args):
    c = connection.cursor()
    args['last_epoch'] = 0
    colnames = '(%s)' % ','.join((k for k in args))
    c.execute("DROP TABLE IF EXISTS options_copy")
    c.execute("CREATE TABLE options_copy %s" % colnames)
    c.execute("SELECT DISTINCT name FROM options")
    names = c.fetchall()
    c.executemany("INSERT INTO options_copy (name) VALUES (?)", names)
    for name, in names:
        data, cols = find_info(name, '*', get_col_names=True)
        print(name)
        print(cols)
        for i, col in enumerate(cols):
            t = type(args[col])
            d = data[i]
            try:
                if t == bool:
                    d = int(bool(d))
                elif t == float:
                    d = float(d)
                elif t == int:
                    d = int(d)
                elif t == list:
                    d = str(d)
                elif t == str:
                    d = str(d)
                else:
                    raise ValueError('Type %s not handled for element %s from column %s' % (str(t), str(data[i]), col))
            except ValueError:
                print(d, col, t)
                raise ValueError
            c.execute("UPDATE options_copy SET %s = ? WHERE name = ?" % col, (d, name))
    connection.commit()

def copy_from_saved():
    c = connection.cursor()
    c.execute("DROP TABLE options")
    c.execute("ALTER TABLE options_copy RENAME TO options")
    connection.commit()


if __name__ == '__main__':
    connect("./checkpoints")
    c = connection.cursor()
    # for f in glob.iglob('./checkpoints/cloudcnn/*'):
    #     print(f)
    #     _add_experiment(f)
    # c.execute("DELETE FROM options WHERE name = 'cloudcnn/2019-06-19_21:16'")
    #
    # connection.commit()
    # c.execute("DELETE FROM options WHERE name = 'cloudcnn/2019-05-28_12:34'")
    # connection.commit()
    # # c.execute('DELETE FROM options WHERE name = ""')
    # # c.execute("UPDATE options SET last_epoch = 130 WHERE name = 'cloudcnn/2019-06-01_11:13'")
    # # for row in c.execute('SELECT n_points, beta, sampling, criterion, name, last_epoch FROM options ORDER BY name'):
    # #     print(row)
    # # print(find_info('cloudcnn/2019-05-28_09:09', '*', True))
    # c.execute("UPDATE options SET beta = 1 WHERE beta = '1.0'")
    for row in c.execute("""
        SELECT * FROM test ORDER BY name
    """):
        print(row)
    # connection.commit()
    close()
