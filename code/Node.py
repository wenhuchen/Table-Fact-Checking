import pandas
import numpy
import copy
from APIs import APIs


class Node(object):
    def __init__(self, rows, memory_str, memory_num, header_str, header_num, must_have, must_not_have):
        # For intermediate results
        self.memory_str = memory_str
        self.memory_num = memory_num
        self.memory_bool = []
        self.header_str = header_str
        self.header_num = header_num
        self.trace_str = [v for k, v in memory_str]
        self.trace_num = [v for k, v in memory_num]
        # For intermediate data frame
        self.rows = [("all_rows", rows)]

        self.cur_str = ""
        self.cur_strs = []
        self.cur_funcs = []

        self.must_have = must_have
        self.must_not_have = must_not_have

        self.row_counter = [1]
        #self.str_counter = [0] * len(memory_str)
        #self.num_counter = [0] * len(memory_num)

    def done(self):
        if self.memory_str_len == 0 and self.memory_num_len == 0 and \
                self.memory_bool_len == 0 and all([_ > 0 for _ in self.row_counter]):
            for funcs in self.must_have:
                if any([f in self.cur_funcs for f in funcs]):
                    continue
                else:
                    return False
            return True
        else:
            return False

    @property
    def tostring(self):
        print("memory_str:", self.memory_str)
        print("memory_num:", self.memory_num)
        print("header_str:", self.header_str)
        print("header_num:", self.header_num)
        print("trace:", self.cur_str)

    def concat(self, new_str, k):
        """
        if APIs[k]['append']:
            if self.cur_str:
                self.cur_str += ";" + new_str
            else:
                self.cur_str = new_str
        else:
            pass
        """
        func = new_str.split('(')[0]
        self.cur_funcs.append(func)
        self.cur_strs.append(new_str)
        # if func == 'max':
        #    self.must_not_have.extend(['max', 'argmax'])
        # if func == 'min':
        #    self.must_not_have.extend(['min', 'argmin'])

    def exist(self, command):
        return command in self.cur_strs

    def clone(self, command, k):
        tmp = copy.deepcopy(self)
        tmp.concat(command, k)
        return tmp

    @property
    def memory_str_len(self):
        return len(self.memory_str)

    @property
    def memory_num_len(self):
        return len(self.memory_num)

    @property
    def tmp_memory_num_len(self):
        return len([_ for _ in self.memory_num if "tmp_" in _ and _ != "tmp_none"])
        # return len(self.memory_num)

    @property
    def tmp_memory_str_len(self):
        return len([_ for _ in self.memory_str if "tmp_" in _])

    @property
    def memory_bool_len(self):
        return len(self.memory_bool)

    @property
    def row_num(self):
        return len(self.rows) - 1

    @property
    def hash(self):
        return hash(frozenset(self.cur_strs))
    """
    cache_hash = hash(tuple(self.memory_str + self.memory_num + self.memory_bool \
                            + self.header_str + self.header_num))
    if self.row_num:
        r = []
        for row in self.rows[1:]:
            r.append(len(row))
            r.append(row.iloc[0][0])
        row_hash = hash(tuple(r))
        return cache_hash + row_hash
    else:
        return cache_hash
    """

    def append_result(self, command, r):
        self.cur_str = "{}={}".format(command, r)

    def append_bool(self, r):
        if self.cur_str != "":
            self.cur_str += ";{}".format(r)
        else:
            self.cur_str = "{}".format(r)

    def get_memory_str(self, i):
        return self.memory_str[i][1]

    def get_memory_num(self, i):
        return self.memory_num[i][1]

    def add_memory_num(self, header, val, command):
        if type(val) == type(1) or type(val) == type(1.2):
            val = val
        else:
            val = val.item()

        self.memory_num.append((header, val))
        self.trace_num.append(command)

    def add_memory_bool(self, header, val):
        if isinstance(val, bool):
            self.memory_bool.append((header, val))
        else:
            raise ValueError("type error: {}".format(type(val)))

    def add_memory_str(self, header, val, command):
        if isinstance(val, str):
            self.memory_str.append((header, val))
            self.trace_str.append(command)
        else:
            raise ValueError("type error: {}".format(type(val)))

    def add_header_str(self, header):
        self.header_str.append(header)

    def add_header_num(self, header):
        self.header_num.append(header)

    def add_rows(self, header, val):
        if isinstance(val, pandas.DataFrame):
            # for row_h, row in self.rows:
            #    if len(row) == len(val) and row.iloc[0][0] == val.iloc[0][0]:
            #        return
            if any([row_h == header for row_h, row in self.rows]):
                return
            self.rows.append((header, val.reset_index(drop=True)))
            self.row_counter.append(0)
        else:
            raise ValueError("type error")

    def inc_row_counter(self, i):
        self.row_counter[i] += 1

    def delete_memory_num(self, *args):
        for i, arg in enumerate(args):
            del self.memory_num[arg - i]
            del self.trace_num[arg - i]

    def delete_memory_str(self, *args):
        for i, arg in enumerate(args):
            del self.memory_str[arg - i]
            del self.trace_str[arg - i]

    def delete_memory_bool(self, *args):
        for i, arg in enumerate(args):
            del self.memory_bool[arg - i]

    def check(self, *args):
        final = {}
        for arg in args:
            if arg == 'row':
                continue

            if arg == ['header_str', 'string']:
                if any([k is not None for k, v in self.memory_str]):
                    continue
                else:
                    return False

            if arg == ['header_num', 'number']:
                if any([k is not None for k, v in self.memory_num]):
                    continue
                else:
                    return False

            if arg == 'string':
                if len(self.memory_str) > 0:
                    continue
                else:
                    return False

            if arg == 'number':
                if len(self.memory_num) > 0:
                    continue
                else:
                    return False

            if arg == 'header_str':
                if len(self.header_str) > 0:
                    continue
                else:
                    return False

            if arg == 'header_num':
                if len(self.header_num) > 0:
                    continue
                else:
                    return False
        return True
