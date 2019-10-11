from APIs import *
from Node import Node
import time
from functools import wraps

# prunning tricks


def dynamic_programming(name, t, orig_sent, sent, tags, mem_str, mem_num, head_str, head_num, label, num=6, debug=False):
    must_have = []
    must_not_have = []
    for k, v in non_triggers.items():
        if isinstance(v[0], list):
            flags = []
            for v_sub in v:
                flag = False
                for trigger in v_sub:
                    if trigger in ['RBR', 'RBS', 'JJR', 'JJS']:
                        if trigger in tags:
                            flag = True
                            break
                    else:
                        if " " + trigger + " " in " " + sent + " ":
                            flag = True
                            break
                flags.append(flag)
            if not all(flags):
                must_not_have.append(k)
        else:
            flag = False
            for trigger in v:
                if trigger in ['RBR', 'RBS', 'JJR', 'JJS']:
                    if trigger in tags:
                        flag = True
                        break
                else:
                    if " " + trigger + " " in " " + sent + " ":
                        flag = True
                        break
            if not flag:
                must_not_have.append(k)

    node = Node(memory_str=mem_str, memory_num=mem_num, rows=t,
                header_str=head_str, header_num=head_num, must_have=must_have, must_not_have=must_not_have)

    count_all = False
    for k, v in mem_num:
        if k == "tmp_input":
            count_all = True
            break

    start_time = time.time()
    # The result storage
    finished = []
    hist = [[node]] + [[] for _ in range(num)]
    cache = {}

    def call(command, f, *args):
        if command not in cache:
            cache[command] = f(*args)
            return cache[command]
        else:
            return cache[command]

    start_time = time.time()
    for step in range(len(hist) - 1):
        # Iterate over father nodes
        saved_hash = []

        def conditional_add(tmp, path):
            if tmp.hash not in saved_hash:
                path.append(tmp)
                saved_hash.append(tmp.hash)

        for root in hist[step]:
            # Iterate over API
            for k, v in APIs.items():
                # propose candidates
                if k in root.must_not_have or not root.check(*v['argument']):
                    continue

                if v['output'] == 'row' and root.row_num >= 2:
                    continue

                if v['output'] == 'num' and root.tmp_memory_num_len >= 3:
                    continue
                if v['output'] == 'str' and root.tmp_memory_str_len >= 3:
                    continue

                if v['output'] == 'bool' and root.memory_bool_len >= 3:
                    continue

                if 'inc_' in k and 'inc' in root.cur_funcs:
                    continue
                """
                elif v['argument'] == ["header_num"]:
                    for l in range(len(root.header_num)):
                        command = v['tostr'](root.header_num[l])
                        if not root.exist(command):
                            tmp = root.clone(command)
                            returned = v['function'](root.header_num[l])
                            tmp.add_header_num(returned)
                            conditional_add(tmp, hist[i + 1])

                elif v['argument'] == ["header_str"]:
                    for l in range(len(root.header_str)):
                        command = v['tostr'](root.header_str[l])
                        if not root.exist(command):
                            tmp = root.clone(command)
                            returned = v['function'](root.header_str[l])
                            tmp.add_header_str(returned)
                            conditional_add(tmp, hist[i + 1])
                """
                # Incrementing/Decrementing/Whether is zero
                if v['argument'] == ["num"]:
                    for i, (h, va) in enumerate(root.memory_num):
                        if v['output'] == 'num':
                            if step == 0 and "tmp" in h:
                                command = v['tostr'](root.trace_num[i])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    returned = call(command, v['function'], va)
                                    tmp.add_memory_num(h, returned, returned)
                                    conditional_add(tmp, hist[step + 1])
                        elif v['output'] == 'bool':
                            if "tmp_" in h and "count" not in h:
                                command = v['tostr'](root.trace_num[i])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    returned = call(command, v['function'], va)
                                    tmp.delete_memory_num(i)
                                    if tmp.done():
                                        tmp.append_result(command, returned)
                                        finished.append((tmp, returned))
                                    else:
                                        tmp.add_memory_bool(command, returned)
                                        conditional_add(tmp, hist[step + 1])
                        elif v['output'] == 'none':
                            if step == 0 and "tmp" in h:
                                command = v['tostr'](root.trace_num[i])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.delete_memory_num(i)
                                    if tmp.done():
                                        continue
                                    else:
                                        conditional_add(tmp, hist[step + 1])
                        else:
                            raise ValueError("Returned Type Wrong")

                # Incrementing/Decrementing/Whether is none
                elif v['argument'] == ["str"]:
                    for i, (h, va) in enumerate(root.memory_str):
                        if v['output'] == 'str':
                            if step == 0:
                                if "tmp_" not in h:
                                    command = v['tostr'](root.trace_str[i])
                                    if not root.exist(command):
                                        tmp = root.clone(command, k)
                                        returned = call(command, v['function'], va)
                                        tmp.add_memory_str(h, returned, returned)
                                        conditional_add(tmp, hist[step + 1])
                        elif v['output'] == 'bool':
                            if k == "existing" and step == 0:
                                pass
                            elif k == "none" and "tmp_" in h:
                                pass
                            else:
                                continue
                            command = v['tostr'](root.trace_str[i])
                            if not root.exist(command):
                                tmp = root.clone(command, k)
                                returned = call(command, v['function'], va)
                                tmp.delete_memory_str(i)
                                if tmp.done():
                                    tmp.append_result(command, returned)
                                    finished.append((tmp, returned))
                                else:
                                    tmp.add_memory_bool(command, returned)
                                    conditional_add(tmp, hist[step + 1])
                        else:
                            raise ValueError("Returned Type Wrong")

                elif v['argument'] == ['row', 'header_str', 'str']:
                    for j, (row_h, row) in enumerate(root.rows):
                        for i, (h, va) in enumerate(root.memory_str):
                            if "tmp_" in h or len(row) == 1:
                                continue
                            for head in root.header_str:
                                if "; " + head + ";" in row_h:
                                    continue
                                command = v['tostr'](row_h, head, root.trace_str[i])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    returned = call(command, v['function'], row, head, va)
                                    if v['output'] == "bool":
                                        tmp.inc_row_counter(j)
                                        tmp.delete_memory_str(i)
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError("Returned Type Wrong")

                elif v['argument'] == ['row', 'header_num', 'num']:
                    for j, (row_h, row) in enumerate(root.rows):
                        for i, (h, va) in enumerate(root.memory_num):
                            if "tmp_" in h or len(row) == 1:
                                continue
                            for head in root.header_num:
                                if "; " + head + ";" in row_h:
                                    continue
                                command = v['tostr'](row_h, head, root.trace_num[i])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    returned = call(command, v['function'], row, head, va)
                                    if v['output'] == "bool":
                                        tmp.inc_row_counter(j)
                                        tmp.delete_memory_num(i)
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError("Returned Type Wrong")

                elif v['argument'] == ['bool', 'bool']:
                    if root.memory_bool_len < 2:
                        continue
                    else:
                        for l in range(0, root.memory_bool_len - 1):
                            for m in range(l + 1, root.memory_bool_len):
                                command = v['tostr'](root.memory_bool[l][0], root.memory_bool[m][0])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    returned = call(command, v['function'], root.memory_bool[l]
                                                    [1], root.memory_bool[m][1])
                                    if v['output'] == "bool":
                                        tmp.delete_memory_bool(l, m)
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError("Returned Type Wrong")

                elif v['argument'] == ['row']:
                    for j, (row_h, row) in enumerate(root.rows):
                        if k == "count":
                            if row_h.startswith('filter'):
                                pass
                            elif row_h == "all_rows":
                                if count_all:
                                    pass
                                else:
                                    continue
                        elif k == "only":
                            if not row_h.startswith('filter'):
                                continue
                        else:
                            if not row_h == "all_rows":
                                continue
                        command = v['tostr'](row_h)
                        if not root.exist(command):
                            tmp = root.clone(command, k)
                            tmp.inc_row_counter(j)
                            returned = call(command, v['function'], row)
                            if v['output'] == 'num':
                                tmp.add_memory_num("tmp_count", returned, command)
                            elif v['output'] == 'row':
                                tmp.add_rows(command, returned)
                                conditional_add(tmp, hist[step + 1])
                            elif v['output'] == 'bool':
                                if tmp.done():
                                    tmp.append_result(command, returned)
                                    finished.append((tmp, returned))
                                elif tmp.memory_bool_len < 2:
                                    tmp.add_memory_bool(command, returned)
                                    conditional_add(tmp, hist[step + 1])
                            else:
                                raise ValueError("error, out of scope")
                            conditional_add(tmp, hist[step + 1])

                elif v['argument'] == ['row', 'row', 'row']:
                    if len(root.rows) < 3:
                        continue
                    _, all_rows = root.rows[0]
                    for i in range(1, len(root.rows) - 1):
                        for j in range(i + 1, len(root.rows)):
                            if v['output'] == 'bool':
                                if len(root.rows[i][1]) != 1 or len(root.rows[j][1]) != 1:
                                    continue
                                command = v['tostr'](root.rows[i][0], root.rows[j][0])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(i)
                                    tmp.inc_row_counter(j)
                                    returned = call(command, v['function'], all_rows, root.rows[i][1], root.rows[j][1])
                                    if returned is not None:
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                            else:
                                raise ValueError("error, out of scope")

                elif v['argument'] == ['row', 'row']:
                    if len(root.rows) < 2:
                        continue
                    for i in range(len(root.rows) - 1):
                        for j in range(i + 1, len(root.rows)):
                            if v['output'] == 'bool':
                                if len(root.rows[i][1]) != 1 and len(root.rows[j][1]) != 1:
                                    continue
                                command = v['tostr'](root.rows[i][0], root.rows[j][0])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(i)
                                    tmp.inc_row_counter(j)
                                    returned = call(command, v['function'], root.rows[i][1], root.rows[j][1])
                                    if returned is not None:
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                            else:
                                raise ValueError("error, out of scope")

                elif v['argument'] == ['row', 'header_num']:
                    if "hop" in k:
                        for j, (row_h, row) in enumerate(root.rows):
                            if len(row) != 1:
                                continue
                            for l in range(len(root.header_num)):
                                command = v['tostr'](row_h, root.header_num[l])
                                if "; " + root.header_num[l] + ";" in row_h:
                                    continue
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    returned = call(command, v['function'], row, root.header_num[l])
                                    if v['output'] == 'num':
                                        tmp.add_memory_num("tmp_" + root.header_num[l], returned, command)
                                        conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError("error, output of scope")
                    else:
                        for j, (row_h, row) in enumerate(root.rows):
                            if len(row) == 1:
                                continue
                            for l in range(len(root.header_num)):
                                command = v['tostr'](row_h, root.header_num[l])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    returned = call(command, v['function'], row, root.header_num[l])
                                    if v['output'] == 'num':
                                        tmp.add_memory_num("tmp_" + root.header_num[l], returned, command)
                                        conditional_add(tmp, hist[step + 1])
                                    elif v['output'] == 'row':
                                        if len(returned) > 0:
                                            tmp.add_rows(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                        else:
                                            continue
                                    else:
                                        raise ValueError("error, output of scope")

                elif v['argument'] == ['row', 'header_str']:
                    if "most_freq" in k:
                        row_h, row = root.rows[0]
                        for l in range(len(root.header_str)):
                            command = v['tostr'](row_h, root.header_str[l])
                            if not root.exist(command):
                                tmp = root.clone(command, k)
                                returned = call(command, v['function'], row, root.header_str[l])
                                if v['output'] == 'str':
                                    if returned is not None:
                                        tmp.add_memory_str("tmp_" + root.header_str[l], returned, command)
                                        conditional_add(tmp, hist[step + 1])
                                else:
                                    raise ValueError("error, output of scope")
                    elif "hop" in k:
                        for j, (row_h, row) in enumerate(root.rows):
                            if len(row) != 1:
                                continue
                            for l in range(len(root.header_str)):
                                if "; " + root.header_str[l] + ";" in row_h:
                                    continue
                                command = v['tostr'](row_h, root.header_str[l])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    returned = call(command, v['function'], row, root.header_str[l])
                                    if v['output'] == 'str':
                                        if isinstance(returned, str):
                                            tmp.add_memory_str("tmp_" + root.header_str[l], returned, command)
                                            conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError("error, output of scope")
                    else:
                        for j, (row_h, row) in enumerate(root.rows):
                            if len(row) == 1:
                                continue
                            for l in range(len(root.header_str)):
                                command = v['tostr'](row_h, root.header_str[l])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    returned = call(command, v['function'], row, root.header_str[l])
                                    if v['output'] == 'str':
                                        if isinstance(returned, str):
                                            tmp.add_memory_str("tmp_" + root.header_str[l], returned, command)
                                            conditional_add(tmp, hist[step + 1])
                                    elif v['output'] == 'row':
                                        if len(returned) > 0:
                                            tmp.add_rows(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                        else:
                                            continue
                                    elif v['output'] == 'num':
                                        tmp.add_memory_num("tmp_count", returned, command)
                                        conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError("error, output of scope")

                elif v['argument'] == ['num', 'num']:
                    if root.memory_num_len < 2:
                        continue
                    for l in range(0, root.memory_num_len - 1):
                        for m in range(l + 1, root.memory_num_len):
                            if 'tmp_' in root.memory_num[l][0] or 'tmp_' in root.memory_num[m][0]:
                                if ("tmp_input" == root.memory_num[l][0] and "tmp_" not in root.memory_num[m][0]) or \
                                        ("tmp_input" == root.memory_num[m][0] and "tmp_" not in root.memory_num[l][0]):
                                    continue
                                elif root.memory_num[l][0] == root.memory_num[m][0] == "tmp_input":
                                    continue
                            else:
                                continue

                            type_l = root.memory_num[l][0].replace('tmp_', '')
                            type_m = root.memory_num[m][0].replace('tmp_', '')
                            if v['output'] == 'num':
                                if type_l == type_m:
                                    command = v['tostr'](root.trace_num[l], root.trace_num[m])
                                    tmp = root.clone(command, k)
                                    tmp.delete_memory_num(l, m)
                                    returned = call(command, v['function'],
                                                    root.get_memory_num(l), root.get_memory_num(m))
                                    tmp.add_memory_num("tmp_" + root.memory_num[l][0], returned, command)
                                    conditional_add(tmp, hist[step + 1])
                            elif v['output'] == 'bool':
                                if type_l == type_m or (type_l == "input" or type_m == "input"):
                                    pass
                                else:
                                    continue

                                if type_l == "count" and type_m == "input" or type_m == "count" and type_l == "input":
                                    if max(root.get_memory_num(l), root.get_memory_num(m)) > len(root.rows[0][1]):
                                        continue

                                command = v['tostr'](root.trace_num[l], root.trace_num[m])
                                tmp = root.clone(command, k)
                                tmp.delete_memory_num(l, m)
                                returned = call(command, v['function'], root.get_memory_num(l), root.get_memory_num(m))
                                if tmp.done():
                                    tmp.append_result(command, returned)
                                    finished.append((tmp, returned))
                                elif tmp.memory_bool_len < 2:
                                    tmp.add_memory_bool(command, returned)
                                    conditional_add(tmp, hist[step + 1])
                            else:
                                raise ValueError("error, output of scope")

                elif v['argument'] == ['str', 'str']:
                    if root.memory_str_len < 2:
                        continue
                    for l in range(0, root.memory_str_len - 1):
                        for m in range(l + 1, root.memory_str_len):
                            if 'tmp_' not in root.memory_str[l][0] and 'tmp_' not in root.memory_str[m][0]:
                                continue
                            type_l = root.memory_str[l][0].replace('tmp_', '')
                            type_m = root.memory_str[m][0].replace('tmp_', '')
                            if type_l == type_m:
                                command = v['tostr'](root.trace_str[l], root.trace_str[m])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.delete_memory_str(l, m)
                                    if v['output'] == 'bool':
                                        returned = call(command, v['function'],
                                                        root.get_memory_str(m), root.get_memory_str(l))
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError("error, output of scope")

                elif v['argument'] == ['row', ['header_str', 'str']]:
                    for j, (row_h, row) in enumerate(root.rows):
                        for i, (h, va) in enumerate(root.memory_str):
                            if "tmp_" not in h:
                                command = v['tostr'](row_h, h, root.trace_str[i])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    tmp.delete_memory_str(tmp.memory_str.index((h, va)))
                                    returned = call(command, v['function'], row, h, va)
                                    if v['output'] == 'row':
                                        if len(returned) > 0:
                                            tmp.add_rows(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                        else:
                                            continue
                                    elif v['output'] == 'bool':
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError('error, output of scope')

                elif v['argument'] == ['row', ['header_num', 'num']]:
                    for j, (row_h, row) in enumerate(root.rows):
                        for i, (h, va) in enumerate(root.memory_num):
                            if "tmp_" not in h:
                                command = v['tostr'](row_h, h, root.trace_num[i])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    tmp.delete_memory_num(tmp.memory_num.index((h, va)))
                                    returned = call(command, v['function'], row, h, va)
                                    if v['output'] == 'row':
                                        if len(returned) > 0:
                                            tmp.add_rows(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                        else:
                                            continue
                                    elif v['output'] == 'bool':
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError('error, output of scope')

                elif v['argument'] == [['header_str', 'str'], ['header_num', 'num']]:
                    if not root.memory_str_len or not root.memory_num_len:
                        continue
                    row_h, row = root.rows[0]
                    for i, (h1, va1) in enumerate(root.memory_str):
                        for j, (h2, va2) in enumerate(root.memory_num):
                            if "tmp_" not in h1 and "tmp_" not in h2:
                                command = v['tostr'](h1, root.trace_str[i], h2, root.trace_num[j])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.delete_memory_str(i)
                                    tmp.delete_memory_num(j)
                                    returned = call(command, v['function'], row, h1, va1, h2, va2)
                                    if v['output'] == 'bool':
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError('error, output of scope')

                elif v['argument'] == [['header_str', 'str'], ['header_str', 'str']]:
                    if root.memory_str_len < 2:
                        continue
                    row_h, row = root.rows[0]
                    for l in range(len(root.memory_str) - 1):
                        for m in range(l + 1, len(root.memory_str)):
                            h1, va1 = root.memory_str[l]
                            h2, va2 = root.memory_str[m]
                            if "tmp_" not in h1 and "tmp_" not in h2 and (h1 != h2):
                                command = v['tostr'](h1, root.trace_str[l], h2, root.trace_str[m])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.delete_memory_str(l, m)
                                    returned = call(command, v['function'], row, h1, va1, h2, va2)
                                    if v['output'] == 'bool':
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                                    else:
                                        raise ValueError('error, output of scope')

                elif v['argument'] == [['header_num', 'num'], ['header_num', 'num']]:
                    if root.memory_num_len < 2:
                        continue
                    row_h, row = root.rows[0]
                    for l in range(len(root.memory_num) - 1):
                        for m in range(l + 1, len(root.memory_num)):
                            h1, va1 = root.memory_num[l]
                            h2, va2 = root.memory_num[m]
                            if ("tmp_" not in h1 and "tmp_" not in h2) and (h1 != h2):
                                command = v['tostr'](h1, root.trace_num[l], h2, root.trace_num[m])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.delete_memory_num(l, m)
                                    returned = call(command, v['function'], row, h1, va1, h2, va2)
                                    if v['output'] == 'bool':
                                        if tmp.done():
                                            tmp.append_result(command, returned)
                                            finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[step + 1])
                else:
                    raise ValueError(k + ": error")

        if len(finished) > 100 or time.time() - start_time > 40:
            break
            # return (name, orig_sent, label, [_[0].cur_str for _ in finished])

    """
    if debug:
        with open('/tmp/results.txt', 'w') as f:
            for h in hist[-1]:
                print(h.cur_strs, file=f)
    """
    return (name, orig_sent, sent, label, [_[0].cur_str for _ in finished])

    # for _ in finished:
    #    print(_[0].cur_str, _[1])
