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
