from APIs import *
from Node import Node
import time
from functools import wraps

# prunning tricks
def dynamic_programming(name, t, orig_sent, sent, tags, mem_str, mem_num, head_str, head_num, label, num=7):
    must_have = []
    must_not_have = []
    #for k, v in triggers.iteritems():
    #    if k in sent and v not in must_have:
    #        must_have.append(v)
    for k, v in non_triggers.iteritems():
        if isinstance(v[0], list):
            flags = []
            for v_sub in v:
                flag = False
                for trigger in v_sub:
                    if trigger in ['RBR', 'JJR', 'JJR', 'JJS']:
                        if trigger in tags:
                            flag = True
                            break
                    else:
                        if trigger in sent:
                            flag = True
                            break
                flags.append(flag)
            if not all(flags):
                must_not_have.append(k)
        else: 
            flag = False
            for trigger in v:
                if trigger in ['RBR', 'JJR', 'JJR', 'JJS']:
                    if trigger in tags:
                        flag = True
                        break
                else:
                    if trigger in sent:
                        flag = True
                        break
            if not flag:
                must_not_have.append(k)

    #print "must have: ", must_have
    #print "Must not have: ", must_not_have
    #print "Valid functions: ", [_ for _ in APIs if _ not in must_not_have]
    node = Node(memory_str=mem_str, memory_num=mem_num, rows=t, 
                header_str=head_str, header_num=head_num, must_have=must_have, must_not_have=must_not_have)

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
    for i in range(len(hist) - 1):
        # Iterate over father nodes
        saved_hash = []
        def conditional_add(tmp, path):
            if tmp.hash not in saved_hash:
                path.append(tmp)
                saved_hash.append(tmp.hash)

        for root in hist[i]:
            # Iterate over API
            for k, v in APIs.iteritems():
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
                # Incrementing
                if v['argument'] == ["num"]:
                    for h, va in root.memory_num:
                        if v['output'] == 'num':
                            if "tmp_" not in h:
                                command = v['tostr'](va)
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    returned = call(command, v['function'], va)
                                    tmp.add_memory_num(h, returned)
                                    conditional_add(tmp, hist[i + 1])
                        elif v['output'] == 'none':
                            if i == 0:
                                command = v['tostr'](va)
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    returned = call(command, v['function'], va)
                                    tmp.delete_memory_num(tmp.memory_num.index((h, va)))
                                    conditional_add(tmp, hist[i + 1])
                        else:
                            raise ValueError("Returned Type Wrong")
                
                elif v['argument'] == ["str"]:
                    for h, va in root.memory_str:
                        if v['output'] == 'str':
                            if "tmp_" not in h:
                                command = v['tostr'](va)
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    returned = call(command, v['function'], va)
                                    tmp.add_memory_str(h, returned)
                                    conditional_add(tmp, hist[i + 1])
                        elif v['output'] == 'none':
                            if i == 0:
                                command = v['tostr'](va)
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    returned = call(command, v['function'], va)
                                    tmp.delete_memory_str(tmp.memory_str.index((h, va)))
                                    conditional_add(tmp, hist[i + 1])
                        elif v['output'] == 'bool':
                            if "tmp_" in h:
                                command = v['tostr'](va)
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    returned = call(command, v['function'], va)
                                    if tmp.done():
                                        if i > 0:
                                            tmp.append_bool(command)
                                            tmp.append_result(returned)
                                            finished.append((tmp, returned))
                                    else:
                                        tmp.add_memory_bool(command, returned)
                                        conditional_add(tmp, hist[i + 1])
                        else:
                            raise ValueError("Returned Type Wrong")
                            """
                            elif v['argument'] == ['bool']:
                                for h, va in root.memory_bool:
                                    if h.startswith('not'):
                                        continue
                                    command = v['tostr'](h)
                                    tmp = root.clone(command, k)
                                    returned = call(command, v['function'], va)
                                    tmp.delete_memory_bool(tmp.memory_bool.index((h, va)))
                                    if tmp.done():
                                        finished.append((tmp, returned))
                                    else:
                                        tmp.add_memory_bool(command, returned)
                                        conditional_add(tmp, hist[i + 1])
                            """
                elif v['argument'] == ['row', 'header_str', 'str']:
                    for j, (row_h, row) in enumerate(root.rows):
                        for h, va in root.memory_str:
                            if "tmp_" in h:
                                continue
                            for head in root.header_str:
                                command = v['tostr'](row_h, head, va)
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    returned = call(command, v['function'], row, head, va)
                                    if v['output'] == "bool":
                                        tmp.delete_memory_str(tmp.memory_str.index((h, va)))
                                        if tmp.done():
                                            if i > 0:
                                                tmp.append_bool(command)
                                                tmp.append_result(returned)                                            
                                                finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[i + 1])
                                    else:
                                        raise ValueError("Returned Type Wrong")
                
                elif v['argument'] == ['row', 'header_num', 'num']:
                    for j, (row_h, row) in enumerate(root.rows):
                        for h, va in root.memory_num:
                            if "tmp_" in h:
                                continue
                            for head in root.header_num:
                                command = v['tostr'](row_h, head, va)
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    returned = call(command, v['function'], row, head, va)
                                    if v['output'] == "bool":
                                        tmp.delete_memory_num(tmp.memory_num.index((h, va)))
                                        if tmp.done():
                                            if i > 0:
                                                tmp.append_bool(command)
                                                tmp.append_result(returned)                                            
                                                finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[i + 1])
                                    else:
                                        raise ValueError("Returned Type Wrong")
                
                elif v['argument'] == ['bool', 'bool']:
                    if root.memory_bool_len < 2:
                        continue
                    else:
                        for l in range(0, root.memory_bool_len - 1):
                            for m in range(l+1, root.memory_bool_len):
                                command = v['tostr'](root.memory_bool[l][0], root.memory_bool[m][0])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    returned = call(command, v['function'], root.memory_bool[l][1], root.memory_bool[m][1])
                                    if v['output'] == "bool":
                                        tmp.delete_memory_bool(l, m)
                                        if tmp.done():
                                            if i > 0:
                                                tmp.append_bool(command)
                                                tmp.append_result(returned)                                            
                                                finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[i + 1])
                                    else:
                                        raise ValueError("Returned Type Wrong")

                elif v['argument'] == ['row']:
                    for j, (row_h, row) in enumerate(root.rows):
                        command = v['tostr'](row_h)
                        if not root.exist(command):
                            tmp = root.clone(command, k)
                            tmp.inc_row_counter(j)
                            returned = call(command, v['function'], row)
                            if v['output'] == 'num':
                                tmp.add_memory_num("tmp_count", returned)
                                tmp.append_result(returned)
                            elif v['output'] == 'row':
                                tmp.add_rows(command, returned)
                            else:
                                raise ValueError("error, out of scope")   
                            conditional_add(tmp, hist[i + 1])
                
                elif v['argument'] == ['row', 'row']:
                    _, all_rows = root.rows[0]
                    for j, (row_h, row) in enumerate(root.rows):
                        if len(row) != 1:
                            continue
                        command = v['tostr'](row_h)
                        if not root.exist(command):
                            tmp = root.clone(command, k)
                            tmp.inc_row_counter(j)
                            returned = call(command, v['function'], all_rows, row)
                            if v['output'] == 'row':
                                if returned is not None:
                                    tmp.add_rows(command, returned)              
                                    conditional_add(tmp, hist[i + 1])
                            elif v['output'] == 'num':
                                tmp.add_memory_num("tmp_none", returned)
                                tmp.append_result(returned)
                                conditional_add(tmp, hist[i + 1])
                            else:
                                raise ValueError("error, out of scope")

                elif v['argument'] == ['row', 'header_num']:
                    if "hop" in k:
                        for j, (row_h, row) in enumerate(root.rows):
                            if len(row) != 1:
                                continue
                            for l in range(len(root.header_num)):
                                command = v['tostr'](row_h, root.header_num[l])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    #tmp.delete_memory_num(l)
                                    returned = call(command, v['function'], row, root.header_num[l])
                                    if v['output'] == 'num':
                                        tmp.add_memory_num("tmp_" + root.header_num[l], returned)
                                        tmp.append_result(returned)
                                        conditional_add(tmp, hist[i + 1])
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
                                    #tmp.delete_memory_num(l)
                                    returned = call(command, v['function'], row, root.header_num[l])
                                    if v['output'] == 'num':
                                        tmp.add_memory_num("tmp_" + root.header_num[l], returned)
                                        tmp.delete_header_num(l)
                                        tmp.append_result(returned)
                                        conditional_add(tmp, hist[i + 1])
                                    elif v['output'] == 'row':
                                        if len(returned) > 0:
                                            tmp.add_rows(command, returned)
                                            conditional_add(tmp, hist[i + 1])
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
                                        tmp.add_memory_str("tmp_" + root.header_str[l], returned)
                                        tmp.append_result(returned)
                                        conditional_add(tmp, hist[i + 1])
                                else:
                                    raise ValueError("error, output of scope")
                    elif "hop" in k:
                        for j, (row_h, row) in enumerate(root.rows):
                            if len(row) != 1:
                                continue
                            for l in range(len(root.header_str)):
                                command = v['tostr'](row_h, root.header_str[l])
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    returned = call(command, v['function'], row, root.header_str[l])
                                    if v['output'] == 'str':
                                        if isinstance(returned, unicode):
                                            tmp.add_memory_str("tmp_" + root.header_str[l], returned)
                                            tmp.append_result(returned)
                                            conditional_add(tmp, hist[i + 1])
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
                                        if isinstance(returned, unicode):
                                            tmp.add_memory_str("tmp_" + root.header_str[l], v['function'](row, root.header_str[l]))
                                            tmp.append_result(returned)
                                            conditional_add(tmp, hist[i + 1])
                                    elif v['output'] == 'row':
                                        if len(returned) > 0:
                                            tmp.add_rows(command, returned)
                                            conditional_add(tmp, hist[i + 1])
                                        else:
                                            continue
                                    else:
                                        raise ValueError("error, output of scope")
                            
                elif v['argument'] == ['num', 'num']:
                    if root.memory_num_len < 2:
                        continue
                    for l in range(0, root.memory_num_len - 1):
                        for m in range(l + 1, root.memory_num_len):
                            if 'tmp_' in root.memory_num[l][0] or 'tmp_' in root.memory_num[m][0]:
                                pass
                            else:
                                continue
                            type_l = root.memory_num[l][0].replace('tmp_', '')
                            type_m = root.memory_num[m][0].replace('tmp_', '')
                            if v['output'] == 'num':
                                if type_l == type_m:
                                    command = v['tostr'](root.get_memory_num(l), root.get_memory_num(m))
                                    tmp = root.clone(command, k)
                                    tmp.delete_memory_num(l, m)
                                    returned = call(command, v['function'], root.get_memory_num(l), root.get_memory_num(m))
                                    tmp.add_memory_num("tmp_" + type_l, returned)
                                    tmp.append_result(returned)
                                    conditional_add(tmp, hist[i + 1])                                    
                            elif v['output'] == 'bool':
                                if (type_l == type_m and type_l != "input") or (type_l == "none" or type_m == "none"):
                                    command = v['tostr'](root.get_memory_num(l), root.get_memory_num(m))
                                    tmp = root.clone(command, k)
                                    tmp.delete_memory_num(l, m)
                                    returned = call(command, v['function'], root.get_memory_num(l), root.get_memory_num(m))
                                    if tmp.done():
                                        if i > 0:
                                            tmp.append_bool(command)
                                            tmp.append_result(returned)
                                            finished.append((tmp, returned))
                                    elif tmp.memory_bool_len < 2:
                                        tmp.add_memory_bool(command, returned)
                                        conditional_add(tmp, hist[i + 1])
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
                                command = v['tostr'](root.get_memory_str(m), root.get_memory_str(l))
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.delete_memory_str(l, m)
                                    if v['output'] == 'bool':
                                        returned = call(command, v['function'], root.get_memory_str(m), root.get_memory_str(l))
                                        if tmp.done():
                                            if i > 0:
                                                tmp.append_bool(command)
                                                tmp.append_result(returned)                                        
                                                finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[i + 1])                                            
                                    else:
                                        raise ValueError("error, output of scope") 
                
                elif v['argument'] == ['row', ['header_str', 'str']]:
                    for j, (row_h, row) in enumerate(root.rows):
                        #if len(row_h) == 1:
                        #    continue
                        for h, va in root.memory_str:
                            if "tmp_" not in h:
                                command = v['tostr'](row_h, h, va)
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    tmp.delete_memory_str(tmp.memory_str.index((h, va)))
                                    returned = call(command, v['function'], row, h, va)
                                    if v['output'] == 'row':
                                        if len(returned) > 0:
                                            tmp.add_rows(command, returned)
                                            conditional_add(tmp, hist[i + 1])
                                        else:
                                            continue
                                    elif v['output'] == 'bool':
                                        if tmp.done():
                                            if i > 0:
                                                tmp.append_bool(command)
                                                tmp.append_result(returned)
                                                finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[i + 1])
                                    else:
                                        raise ValueError('error, output of scope')
                                    
                elif v['argument'] == ['row', ['header_num', 'num']]:
                    for j, (row_h, row) in enumerate(root.rows):
                        #if len(row) == 1:
                        #    continue
                        for h, va in root.memory_num:
                            if "tmp_" not in h:
                                command = v['tostr'](row_h, h, va)
                                if not root.exist(command):
                                    tmp = root.clone(command, k)
                                    tmp.inc_row_counter(j)
                                    tmp.delete_memory_num(tmp.memory_num.index((h, va)))
                                    returned = call(command, v['function'], row, h, va)
                                    if v['output'] == 'row':
                                        if len(returned) > 0:
                                            tmp.add_rows(command, returned)
                                            conditional_add(tmp, hist[i + 1])
                                        else:
                                            continue
                                    elif v['output'] == 'bool':
                                        if tmp.done():
                                            if i > 0:
                                                tmp.append_bool(command)
                                                tmp.append_result(returned)                                            
                                                finished.append((tmp, returned))
                                        elif tmp.memory_bool_len < 2:
                                            tmp.add_memory_bool(command, returned)
                                            conditional_add(tmp, hist[i + 1])                                           
                                    else:
                                        raise ValueError('error, output of scope')
                else:
                    raise ValueError(k + ": error")
                """
                elif v['argument'] == [['header_str', 'str'], ['header_num', 'num']]:
                    if not root.memory_str_len or not root.memory_num_len:
                        continue
                    row_h, row = root.rows[0]
                    for h1, va1 in root.memory_str:
                        for h2, va2 in root.memory_num:
                            if "tmp_" not in h1 and "tmp_" not in h2:
                                command = v['tostr'](h1, va1, h2, va2)
                                if not root.exist(command):
                                    tmp = root.clone(command)
                                    tmp.delete_memory_str(tmp.memory_str.index((h1, va1)))
                                    tmp.delete_memory_num(tmp.memory_num.index((h2, va2)))
                                    returned = call(command, v['function'], row, h1, va1, h2, va2)
                                    if tmp.done():
                                        finished.append((tmp, returned))
                                    elif tmp.memory_bool_len < 2:
                                        tmp.add_memory_bool(command, returned)
                                        conditional_add(tmp, hist[i + 1])

                elif v['argument'] == [['header_str', 'str'], ['header_str', 'str']]:
                    if root.memory_str_len < 2:
                        continue
                    row_h, row = root.rows[0]
                    for l in range(root.memory_str_len):
                        for m in range(l + 1, root.memory_str_len):
                            if "tmp_" not in root.memory_str[l][0] and "tmp_" not in root.memory_str[m][0] and root.memory_str[l][0] != root.memory_str[m][0]:
                                h1, va1 = root.memory_str[l]
                                h2, va2 = root.memory_str[m]
                                command = v['tostr'](h1, va1, h2, va2)
                                if not root.exist(command):
                                    tmp = root.clone(command)
                                    tmp.delete_memory_str(tmp.memory_str.index((h1, va1)))
                                    tmp.delete_memory_str(tmp.memory_str.index((h2, va2)))
                                    returned = call(command, v['function'], row, h1, va1, h2, va2)
                                    if tmp.done():
                                        finished.append((tmp, returned))
                                    elif tmp.memory_bool_len < 2:
                                        tmp.add_memory_bool(command, returned)
                                        conditional_add(tmp, hist[i + 1])

                elif v['argument'] == [['header_num', 'num'], ['header_num', 'num']]:
                    if root.memory_num_len < 2:
                        continue
                    row_h, row = root.rows[0]
                    for l in range(root.memory_num_len):
                        for m in range(l + 1, root.memory_num_len):
                            if "tmp_" not in root.memory_num[l][0] and "tmp_" not in root.memory_num[m][0] and root.memory_num[l][0] != root.memory_num[m][0]:
                                h1, va1 = root.memory_num[l]
                                h2, va2 = root.memory_num[m]
                                command = v['tostr'](h1, va1, h2, va2)
                                if not root.exist(command):
                                    tmp = root.clone(command)
                                    tmp.delete_memory_num(tmp.memory_num.index((h1, va1)))
                                    tmp.delete_memory_num(tmp.memory_num.index((h2, va2)))
                                    returned = call(command, v['function'], row, h1, va1, h2, va2)
                                    if tmp.done():
                                        finished.append((tmp, returned))
                                    elif tmp.memory_bool_len < 2:
                                        tmp.add_memory_bool(command, returned)
                                        conditional_add(tmp, hist[i + 1])
                """                                                       

        if len(finished) > 100 or time.time() - start_time > 30:
            break

    #print "used time {} to get {} programs".format(time.time() - start_time, len(hist[-1]))
    return (name, orig_sent, label, [_[0].cur_str for _ in finished])
    #with open('/tmp/results.txt', 'w') as f:
    #    for h in hist[-1]:
    #       print >> f, h.cur_str

    #for _ in finished:
    #    print _[0].cur_str, _[1]

    #print "finished {} programs".format(len(finished))
