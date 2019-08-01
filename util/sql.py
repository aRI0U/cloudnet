import numpy as np
import os
import sqlite3
import warnings

class Database():
    # init operations
    def __init__(self, path):
        self.connection = None
        self.c = None
        self.path = path

    def __enter__(self):
        self.connection = sqlite3.connect(os.path.join(self.path, 'options.db'))
        self.c = self.connection.cursor()
        return self

    def __exit__(self, type, value, traceback):
        self.connection.close()

    # BASIC OPERATIONS

    def _exec(self, *args):
        self.c.execute(*args)

    def _new_row(self, table, id):
        self._exec("INSERT INTO %s (id) VALUES (?)" % table, (id,))

    def _update(self, id, col, value):
        self._exec("UPDATE options SET %s = ? WHERE id = ?" % col, (value, id))



    def init_tables(self):
        self.c.executescript("""
            CREATE TABLE IF NOT EXISTS options (
                id INTEGER PRIMARY KEY,
                dataroot TEXT,
                adambeta1 REAL,
                adambeta2 REAL,
                batchSize INTEGER,
                beta REAL,
                checkpoints_dir TEXT,
                criterion TEXT,
                dataset_mode TEXT,
                fineSize INTEGER,
                input_nc INTEGER,
                loadSize INTEGER,
                lr REAL,
                lr_policy TEXT,
                lr_decay_iters INTEGER,
                lstm_hidden_size INTEGER,
                max_dataset_size INTEGER,
                model TEXT,
                name TEXT,
                no_dropout INTEGER,
                no_flip INTEGER,
                n_points INTEGER,
                output_nc INTEGER,
                resize_or_crop INTEGER,
                sampling TEXT,
                serial_batches INTEGER,
                split REAL
            );

            CREATE TABLE IF NOT EXISTS last_epochs (
                id INTEGER,
                epoch INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS branches (
                id INTEGER,
                branch TEXT DEFAULT 'master'
            );

            CREATE TABLE IF NOT EXISTS test (
                id INTEGER,
                epoch INTEGER,
                pos_err REAL,
                ori_err REAL,
                phase TEXT
            );

            CREATE TABLE IF NOT EXISTS tmp (
                id INTEGER,
                epoch INTEGER,
                phase TEXT
            );
        """)

    def drop_tables(self, *args):
        print('This action will definitely delete the following tables: %s' % ', '.join(args))
        if input('Continue? [y/n]').lower() in 'yes':
            for table in args:
                self._exec("DROP TABLE IF EXISTS %s" % table)

    def commit(self):
        self.connection.commit()

    def new_experiment(self, **kwargs):
        self._exec("SELECT id FROM options ORDER BY id")
        ids = self.c.fetchall()
        id = 0
        while id < len(ids) and (id,) == ids[id]:
            id += 1
        self._new_row('options', id)

        for k, v in kwargs.items():
            try:
                self._update(id, k, v)
            except sqlite3.OperationalError:
                continue
            # self._exec('SELECT * FROM opts WHERE id = ?', (id,))
        self._new_row('last_epochs', id)
        self._new_row('branches', id)
        head = os.path.join('.git', 'HEAD')
        if os.path.isfile(head):
            with open(head) as f:
                branch = f.read().split('/')[-1]
                self._exec("UPDATE branches SET branch = ? WHERE id = ?", (branch,id))
        return id

    def find_experiment(self, **kwargs):
        query = "SELECT id FROM options WHERE "
        keys = []
        vals = []
        for k, v in kwargs.items():
            keys.append('%s = ?' % k)
            vals.append(v)
        query += ' AND '.join(keys)
        self._exec(query, tuple(vals))
        # print(query, vals)
        id_list = self.c.fetchall()
        if len(id_list) == 0:
            return None
        if len(id_list) > 1:
            warnings.warn("There are multiple experiments with such options. The most recent one has been loaded.")
        return max(id_list)[0]

    def find_info(self, id, cols, get_col_names=False):
        # for row in self.c.execute("SELECT * FROM last_epochs"):
        #     print(row)
        # self._exec("UPDATE options SET name = 'cloudcnn/2019-07-08_13:34' WHERE id = 93")
        # self._exec("UPDATE options SET name = 'cloudcnn/2019-07-08_13:33' WHERE id = 92")
        # self.commit()
        self._exec("SELECT %s FROM options WHERE id = ?" % cols, (id,))
        if get_col_names:
            description = []
            for d in self.c.description:
                description.append(d[0])
            return self.c.fetchone(), description
        return self.c.fetchone()

    def update_last_epoch(self, id, epoch):
        self._exec("UPDATE last_epochs SET epoch = ? WHERE id = ?", (epoch,id))

    def get_last_epoch(self, id):
        self._exec("SELECT epoch FROM last_epochs WHERE id = ?", (id,))
        return self.c.fetchone()[0]

    ## TEST METHODS

    def test_results(self, id, phase=None, epoch=None):
        query = "SELECT epoch, pos_err, ori_err FROM test WHERE id = ?"
        if phase is None:
            if epoch is None:
                self._exec(query + " ORDER BY epoch", (id,))
            else:
                self._exec(query + " AND epoch = ? ORDER BY epoch", (id,epoch))
        else:
            if epoch is None:
                self._exec(query + " AND phase = ? ORDER BY epoch", (id,phase))
            else:
                self._exec(query + " AND phase = ? AND epoch = ? ORDER BY epoch", (id, phase, epoch))
        res = self.c.fetchall()
        if res is None or len(res) == 0:
            # print('# TODO: warnings')
            return None
        return np.array(res)

    def new_test(self, id, epoch, phase):
        self._exec("INSERT INTO tmp (id, epoch, phase) VALUES (?,?,?)", (id, epoch, phase))

    def add_test_result(self, id, epoch, phase, pos_err, ori_err):
        self._exec("INSERT INTO test (id, epoch, phase, pos_err, ori_err) VALUES (?,?,?,?,?)", (id, epoch, phase, pos_err, ori_err))

    def remove_tmp_test(self, id, epoch, phase):
        self._exec("DELETE FROM tmp WHERE id = ? AND epoch = ? AND phase = ?", (id, epoch, phase))

    def is_test(self, id, epoch, phase):
        self._exec("SELECT id FROM test WHERE id = ? AND epoch = ? AND phase = ?", (id, epoch, phase))
        if self.c.fetchone() is not None:
            return True
        self._exec("SELECT id FROM tmp WHERE id = ? AND epoch = ? AND phase = ?", (id, epoch, phase))
        return self.c.fetchone() is not None











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

def update_lr(lr, name):
    c = connection.cursor()
    c.execute("UPDATE options SET lr = ? WHERE name = ?", (lr, name))
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
    with Database('./checkpoints') as db:
        db._exec("DELETE FROM test WHERE id = 19")
        db.commit()
