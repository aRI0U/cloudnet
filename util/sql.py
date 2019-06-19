import os
import sqlite3

relevant_options = ['batchSize', 'beta', 'criterion', 'dataset_mode','input_nc','max_dataset_size','model','no_dropout','n_points', 'output_nc', 'sampling', 'seed', 'serial_batches']

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
    command = "INSERT INTO options VALUES ("
    for k, v in args.items():
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

def _add_experiment(dir, last_epoch):
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
        command += "%s == '%s'" % (o, eval('str(opt.%s)' % o))
        command += " AND "
    command += "beta == '%s'" % opt.beta
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
    c.execute("CREATE TABLE IF NOT EXISTS test (name, epoch, pos_err, ori_err)")

def new_test(name, epoch):
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
    c.execute("INSERT INTO test (name, epoch) VALUES (?,?)", (name, epoch))
    connection.commit()

def add_test_result(name, epoch, pos_err, ori_err):
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
    c.execute("UPDATE test SET pos_err = ?, ori_err = ? WHERE name = ? AND epoch = ?", (pos_err, ori_err, name, epoch))
    connection.commit()

def get_test_result(name, epoch=None):
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
        c.execute("SELECT epoch, pos_err, ori_err FROM test WHERE name = ? ORDER BY epoch", (name,))
        return c.fetchall()

    c.execute("SELECT pos_err, ori_err FROM test WHERE name = ? AND epoch = ?", (name, epoch))
    return c.fetchone()

def _delete_test(name, epoch):
    c = connection.cursor()
    c.execute("DELETE FROM test WHERE name = ? AND epoch = ?", (name, epoch))
    connection.commit()

def remove_empty_tests():
    c = connection.cursor()
    c.execute("DELETE FROM test WHERE pos_err IS NULL OR ori_err IS NULL")
    connection.commit()


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
        print(name)
        data, cols = find_info(name, '*', get_col_names=True)
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
                elif t == str:
                    d = str(d)
                else:
                    raise ValueError('Type %s not handled for element %s from column %s' % (str(t), str(data[i]), col))
            except ValueError:
                print(d, col, t)
                raise ValueError
            c.execute("UPDATE options_copy SET %s = ? WHERE name = ?" % col, (d, name))
    connection.commit()

if __name__ == '__main__':
    connect("./checkpoints")
    c = connection.cursor()
    for row in c.execute("SELECT * FROM options_copy ORDER BY name"):
        print(row)
    # c.execute('DELETE FROM options WHERE name = ""')
    # c.execute("UPDATE options SET last_epoch = 130 WHERE name = 'cloudcnn/2019-06-01_11:13'")
    # for row in c.execute('SELECT n_points, beta, sampling, criterion, name, last_epoch FROM options ORDER BY name'):
    #     print(row)
    # print(find_info('cloudcnn/2019-05-28_09:09', '*', True))
    # for row in c.execute("""
    #     SELECT * FROM test ORDER BY name
    # """):
    #     print(row)
    # connection.commit()
    close()
