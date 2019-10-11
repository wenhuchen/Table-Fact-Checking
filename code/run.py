# encoding=utf8
import json
import pandas
import numpy
from beam_search import dynamic_programming
from multiprocessing import Pool
import multiprocessing
import sys
import time
import argparse
import os
from APIs import *

parser = argparse.ArgumentParser()
parser.add_argument("--synthesize", default=False, action="store_true", help="whether to synthesize data")
parser.add_argument("--sequential", default=False, action="store_true", help="Whether to use sequential or distributed")
parser.add_argument("--debug", default=False, action="store_true", help="Whether to use debugging mode")
parser.add_argument("--part", type=int, default=0, help="choose a part")
parser.add_argument("--split", type=int, default=1, help="how many splits")
parser.add_argument("--output", type=str, default="../all_programs", help="which folder to store the results")
args = parser.parse_args()

with open('../tokenized_data/full_cleaned.json') as f:
    data = json.load(f)

months_a = ['january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december']
months_b = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']


def isnumber(string):
    return string in [numpy.dtype('int64'), numpy.dtype('int32'), numpy.dtype('float32'), numpy.dtype('float64')]


def list2tuple(inputs):
    mem = []
    for s in inputs:
        mem.append(tuple(s))
    return mem


def split(string, option):
    if option == "row":
        return string.split(',')[0]
    else:
        return string.split(',')[1]


if not args.synthesize:
    count = 0
    preprocessed = []
    for idx, table_name in enumerate(data):
        t = pandas.read_csv('../data/all_csv/{}'.format(table_name), delimiter="#")
        cols = t.columns
        mapping = {i: "num" if isnumber(t) else "str" for i, t in enumerate(t.dtypes)}
        entry = data[table_name]
        caption = entry[3].split(' ')

        for sent, label, pos_tag in zip(entry[0], entry[1], entry[2]):
            count += 1
            inside = False
            position = False
            masked_sent = ''
            position_buf, mention_buf = '', ''
            mem_num, head_num, mem_str, head_str = [], [], [], []
            ent_index = 0
            for n in range(len(sent)):
                if sent[n] == '#':
                    if position:
                        if position_buf.startswith('0'):
                            idx = int(split(position_buf, "col"))
                            if mapping[idx] == 'num':
                                if cols[idx] not in head_num:
                                    head_num.append(cols[idx])
                            else:
                                if cols[idx] not in head_str:
                                    head_str.append(cols[idx])
                        else:
                            row = int(split(position_buf, "row"))
                            idx = int(split(position_buf, "col"))
                            if idx == -1:
                                pass
                            else:
                                if mapping[idx] == 'num':
                                    if mention_buf.isdigit():
                                        mention_buf = int(mention_buf)
                                    else:
                                        try:
                                            mention_buf = float(mention_buf)
                                        except Exception:
                                            import pdb
                                            pdb.set_trace()
                                    val = (cols[idx], mention_buf)
                                    if val not in mem_num:
                                        mem_num.append(val)
                                else:
                                    if len(fuzzy_match(t, cols[idx], mention_buf)) == 0:
                                        val = (cols[idx], mention_buf)
                                    else:
                                        val = (cols[idx], mention_buf)
                                    if val not in mem_str:
                                        mem_str.append(val)
                        masked_sent += "<ENTITY{}>".format(ent_index)
                        ent_index += 1
                        position_buf = ""
                        mention_buf = ""
                        inside = False
                        position = False
                    else:
                        inside = True
                elif sent[n] == ';':
                    position = True
                else:
                    if position:
                        position_buf += sent[n]
                    elif inside:
                        mention_buf += sent[n]
                    else:
                        masked_sent += sent[n]

            tokens = masked_sent.split()
            i = 0
            while i < len(tokens):
                _ = tokens[i]
                if i + 1 < len(tokens):
                    if _.isdigit() and (tokens[i + 1] not in ["thousand", "hundred"]):
                        num = int(_)
                        i += 1
                    elif _.isdigit() and tokens[i + 1] in ["thousand", "hundred"]:
                        if tokens[i + 1] == "thousand":
                            num = int(_) * 1000
                            i += 2
                        elif tokens[i + 1] == "hundred":
                            num = int(_) * 100
                            i += 2
                    elif _ == "a" and tokens[i + 1] in ["thousand", "hundred"]:
                        if tokens[i + 1] == "thousand":
                            num = 1000
                            i += 2
                        elif tokens[i + 1] == "hundred":
                            num = 100
                            i += 2
                    elif '.' in tokens[i]:
                        try:
                            num = float(_)
                            i += 1
                        except Exception:
                            i += 1
                            continue
                    else:
                        i += 1
                        continue
                else:
                    if _.isdigit():
                        num = int(_)
                        i += 1
                    elif '.' in tokens[i]:
                        try:
                            num = float(_)
                            i += 1
                        except Exception:
                            i += 1
                            continue
                    else:
                        i += 1
                        continue

                features = []

                if tokens[i - 2] in months_b + months_a:
                    features.append(-6)
                else:
                    features.append(0)

                if any([_ in tokens for _ in ["than", "over", "more", "less"]]):
                    features.append(2)
                else:
                    features.append(0)

                if any([_ in pos_tag for _ in ["RBR", "JJR"]]):
                    features.append(1)
                else:
                    features.append(0)

                if num > 50:
                    if num > 1900 and num < 2020:
                        features.append(-4)
                    else:
                        features.append(2)
                else:
                    if num > len(t):
                        features.append(2)
                    else:
                        features.append(0)

                if len(head_num) > 0:
                    features.append(1)
                else:
                    features.append(0)

                flag = False
                for h in head_num:
                    if h not in map(lambda x: x[0], mem_num):
                        flag = True

                if flag:
                    features.append(2)
                else:
                    features.append(0)

                if sum(features) >= 3:
                    for h in head_num:
                        if any([_ == h for _ in mem_num]):
                            continue
                        else:
                            mem_num.append((h, num))
                elif sum(features) >= 0:
                    mem_num.append(("tmp_input", num))

            for k, v in mem_num:
                if k not in head_num and k != "tmp_input":
                    head_num.append(k)

            for k, v in mem_str:
                if k not in head_str:
                    head_str.append(k)

            preprocessed.append((table_name, sent, pos_tag, masked_sent, mem_str,
                                 mem_num, head_str, head_num, "nt-{}".format(len(preprocessed)), label))

    length = len(preprocessed) // args.split
    for i in range(args.split):
        with open('../preprocessed_data_program/preprocessed.json'.format(i), 'w') as f:
            if i == args.split - 1:
                json.dump(preprocessed[i * length:], f, indent=2)
            else:
                json.dump(preprocessed[i * length: (i + 1) * length], f, indent=2)

else:
    with open('../preprocessed_data_program/preprocessed.json'.format(args.part), 'r') as f:
        data = json.load(f)

    with open('../data/complex_ids.json') as f:
        complex_ids = json.load(f)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    def func(inputs):
        table_name, sent, pos_tag, masked_sent, mem_str, mem_num, head_str, head_num, idx, labels = inputs
        t = pandas.read_csv('../data/all_csv/{}'.format(table_name), delimiter="#", encoding='utf-8')
        t.fillna('')
        if args.sequential:
            res = dynamic_programming(table_name, t, sent, masked_sent, pos_tag, mem_str,
                                      mem_num, head_str, head_num, labels, 5, debug=True)
            print(idx, res[:-1])
            for r in res[-1]:
                print(r)
        else:
            try:
                if not os.path.exists('{}/{}.json'.format(args.output, idx)):
                    res = dynamic_programming(table_name, t, sent, masked_sent, pos_tag,
                                              mem_str, mem_num, head_str, head_num, labels, 7)
                    with open('{}/{}.json'.format(args.output, idx), 'w') as f:
                        json.dump(res, f, indent=2)
            except Exception:
                print("failed {}, {}".format(table_name, idx))

    table_name = [_[0] for _ in data]
    sent = [_[1] for _ in data]
    pos_tag = [_[2] for _ in data]
    masked_sent = [_[3] for _ in data]
    mem_str = [list2tuple(_[4]) for _ in data]
    mem_num = [list2tuple(_[5]) for _ in data]
    head_str = [_[6] for _ in data]
    head_num = [_[7] for _ in data]
    idxes = [_[8] for _ in data]
    labels = [_[9] for _ in data]

    if args.sequential:
        for arg in zip(table_name, sent, pos_tag, masked_sent, mem_str, mem_num, head_str, head_num, idxes, labels):
            if arg[8] in ["nt-56710"]:
                func(arg)
    else:
        cores = multiprocessing.cpu_count()
        print("Using {} cores".format(cores))
        pool = Pool(cores)
        res = pool.map(func, zip(table_name, sent, pos_tag, masked_sent,
                                 mem_str, mem_num, head_str, head_num, idxes, labels))

        pool.close()
        pool.join()
