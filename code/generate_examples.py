# encoding=utf8
import sys
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pandas
import json
import re
import string
from unidecode import unidecode

reload(sys)
sys.setdefaultencoding('utf8')

#stop_words = ['be', 'she', 'he', 'her', 'his', 'their', 'the', 'it', ',', '.', '-', 'also', 'will', 'would', 'this', 'that',
#             'these', 'those', 'well', 'with', 'on', 'at', 'and', 'as', 'for', 'from', 'in', 'its', 'of', 'to', 'a',
#             'an', 'where', 'when', 'by', 'not', "'s", "'nt", "make", 'who', 'have', 'within', 'without', 'what',
#             'during', 'than', 'then', 'if', 'when', 'while', 'time', 'appear', 'attend', 'every', 'one', 'two', 'over',
#            'both', 'above', 'only', ",", ".", "(", ")", "&", ":"]

#useless_words = [',', '.', "'s"]
with open('../data/stop_words.json') as f:
    stop_words = json.load(f)

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def replace_useless(s):
    s = s.replace(',', '')
    s = s.replace('.', '')
    s = s.replace('/', '')
    s = s.replace('-', '')
    return s

def get_closest(string, indexes, tab):
    dist = 10000
    len_string = len(string.split())
    for index in indexes:
        entity = replace_useless(tab[index[0]][index[1]])
        len_tab = len(entity.split())
        if abs(len_tab - len_string) == 0:
            return index
        elif abs(len_tab - len_string) < dist:
            minimum = index
            dist = abs(len_tab - len_string)

    if dist > len_string * 3:
        return None
    
    return minimum

def replace_number(string):
    string = re.sub(r'(\b)one(\b)', r'\g<1>1\g<2>', string)
    string = re.sub(r'(\b)two(\b)', '\g<1>2\g<2>', string)
    string = re.sub(r'(\b)three(\b)', '\g<1>3\g<2>', string)
    string = re.sub(r'(\b)four(\b)', '\g<1>4\g<2>', string)
    string = re.sub(r'(\b)five(\b)', '\g<1>5\g<2>', string)
    string = re.sub(r'(\b)six(\b)', '\g<1>6\g<2>', string)
    string = re.sub(r'(\b)seven(\b)', '\g<1>7\g<2>', string)
    string = re.sub(r'(\b)eight(\b)', '\g<1>8\g<2>', string)
    string = re.sub(r'(\b)nine(\b)', '\g<1>9\g<2>', string)
    string = re.sub(r'(\b)ten(\b)', '\g<1>10\g<2>', string)
    string = re.sub(r'(\b)eleven(\b)', '\g<1>11\g<2>', string)
    string = re.sub(r'(\b)twelve(\b)', '\g<1>12\g<2>', string)
    string = re.sub(r'(\b)thirteen(\b)', '\g<1>13\g<2>', string)
    string = re.sub(r'(\b)fourteen(\b)', '\g<1>14\g<2>', string)
    string = re.sub(r'(\b)fifteen(\b)', '\g<1>15\g<2>', string)
    string = re.sub(r'(\b)sixteen(\b)', '\g<1>16\g<2>', string)
    string = re.sub(r'(\b)seventeen(\b)', '\g<1>17\g<2>', string)
    string = re.sub(r'(\b)eighteen(\b)', '\g<1>18\g<2>', string)
    string = re.sub(r'(\b)nineteen(\b)', '\g<1>19\g<2>', string)
    string = re.sub(r'(\b)twenty(\b)', '\g<1>20\g<2>', string)
    return string

def replace(w, transliterate):
    if w in transliterate:
        return transliterate[w]
    else:
        return w

def postprocess(inp, backbone, trans_backbone, transliterate, tabs, recover_dict):
    new_str = []
    new_tags = []
    buf = ""
    last = set()
    inp, pos_tags = get_lemmatize(inp, True)
    for w, p in zip(inp, pos_tags):
        if w in backbone:
            if buf == "":
                last = set(backbone[w])
                buf = recover_dict.get(w, w)
            else:
                proposed = set(backbone[w]) & last
                if not proposed:
                    if buf not in stop_words:
                        closest = get_closest(buf, last, tabs)
                        if closest is not None:
                            if closest[0] == 0:
                                buf = '#{};h{}#'.format(buf, closest[1])
                            else:
                                buf = '#{};c{}#'.format(buf, closest[1])
                    new_str.append(buf)
                    new_tags.append('ENT')
                    buf = recover_dict.get(w, w)
                    last = set(backbone[w])
                else:
                    buf += " " + recover_dict.get(w, w)
                    last = proposed
        elif w in trans_backbone and w not in stop_words:
            if buf == "":
                last = set(trans_backbone[w])
                buf = transliterate[w]
            else:
                proposed = set(trans_backbone[w]) & last
                if not proposed:
                    if buf not in stop_words:
                        closest = get_closest(buf, last, tabs)
                        if closest is not None:
                            if closest[0] == 0:
                                buf = '#{};h{}#'.format(buf, closest[1])
                            else:
                                buf = '#{};c{}#'.format(buf, closest[1])
                    new_str.append(buf)
                    new_tags.append('ENT')
                    buf = transliterate[w]
                    last = set(trans_backbone[w])
                else:
                    buf += " " + transliterate[w]
                    last = proposed
        else:
            if buf != "":
                if buf not in stop_words:
                    closest = get_closest(buf, last, tabs)
                    if closest is not None:
                        if closest[0] == 0:
                            buf = '#{};h{}#'.format(buf, closest[1])
                        else:
                            buf = '#{};c{}#'.format(buf, closest[1])
                new_str.append(buf)
                new_tags.append('ENT')
            buf = ""
            last = set()
            new_str.append(replace_number(w))
            new_tags.append(p)
    
    if buf != "":
        if buf not in stop_words:
            closest = get_closest(buf, last, tabs)
            if closest is not None:
                if closest[0] == 0:
                    buf = '#{};h{}#'.format(buf, closest[1])
                else:
                    buf = '#{};c{}#'.format(buf, closest[1])
        new_str.append(buf)
        new_tags.append("ENT")
    return " ".join(new_str), " ".join(new_tags)

def get_lemmatize(words, return_pos, recover_dict=None):
    #words = nltk.word_tokenize(words)
    words = words.strip().split(' ')
    pos_tags = [_[1] for _ in nltk.pos_tag(words)]
    word_roots = []
    for w, p in zip(words, pos_tags):
        if is_ascii(w) and p in tag_dict:
            lemm = lemmatizer.lemmatize(w, tag_dict[p])
            if recover_dict is not None and lemm != w:
                recover_dict[lemm] = w
            word_roots.append(lemm)
        else:
            word_roots.append(w)
    if return_pos:
        return word_roots, pos_tags
    else:
        return word_roots

tag_dict = {"JJ": wordnet.ADJ,
            "NN": wordnet.NOUN,
            "NNS": wordnet.NOUN,
            "NNP": wordnet.NOUN,
            "NNPS": wordnet.NOUN,
            "VB": wordnet.VERB,
            "VBD": wordnet.VERB,
            "VBG": wordnet.VERB,
            "VBN": wordnet.VERB,
            "VBP": wordnet.VERB,
            "VBZ": wordnet.VERB,
            "RB": wordnet.ADV,
            "RP": wordnet.ADV}

lemmatizer = WordNetLemmatizer()

with open('../data/short_subset.txt') as f:
    limit_length = [_.strip() for _ in f.readlines()]

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

debug = False
if debug:
    with open('../READY/r1_training_all.json') as f:
        data = json.load(f)
    backbone = {}
    tabs = []
    count = 0
    table = 'data/all_csv/2-1859269-1.html.csv'
    with open(table) as f:
        for k, _ in enumerate(f.readlines()):
            _ = _.decode('utf8')
            tabs.append([])
            for l, w in enumerate(_.strip().split('#')):
                tabs[-1].append(w)
                if len(w) > 0:
                    w = get_lemmatize(w, False)
                    for sub in w:
                        if sub not in backbone:
                            backbone[sub] = [(k, l)]
                        else:
                            backbone[sub].append((k, l))
    for d1, d2 in zip(*data['2-18540104-2.html.csv']):
        sent, tag = postprocess(d1, backbone, tabs)
        print sent
else:
    def get_func(filename, output):
        with open(filename) as f:
            data = json.load(f)
        r1_results = {}
        count = 0
        for name in data:
            entry = data[name]
            backbone = {}
            trans_backbone = {}
            transliterate = {}
            tabs = []
            recover_dict = {}
            with open('../data/all_csv/' + name, 'r') as f:
                for k, _ in enumerate(f.readlines()):
                    _ = _.decode('utf8')
                    tabs.append([])
                    for l, w in enumerate(_.strip().split('#')):
                        tabs[-1].append(w)
                        if len(w) > 0:
                            w = get_lemmatize(w, False, recover_dict)
                            for sub in w:
                                if sub not in backbone:
                                    backbone[sub] = [(k, l)]
                                    if not is_ascii(sub):
                                        trans_backbone[unidecode(sub)] = [(k, l)]
                                        transliterate[unidecode(sub)] = sub
                                else:
                                    backbone[sub].append((k, l))
                                    if not is_ascii(sub):
                                        trans_backbone[unidecode(sub)].append((k, l))
                                        transliterate[unidecode(sub)] = sub

            for w in entry[2].strip().split(' '):
                if w not in backbone:
                    backbone[w] = [(-1, -1)]
                else:
                    backbone[w].append((-1, -1))
            tabs.append([entry[2].strip()])
            for i in range(len(entry[0])):
                count += 1
                if name in r1_results:
                    sent, tag = postprocess(entry[0][i], backbone, trans_backbone, transliterate, tabs, recover_dict)
                    r1_results[name][0].append(sent)
                    r1_results[name][1].append(entry[1][i])
                    r1_results[name][2].append(tag)
                    #r1_results[name][3].append(entry[2])
                else:
                    sent, tag = postprocess(entry[0][i], backbone, trans_backbone, transliterate, tabs, recover_dict)
                    r1_results[name] = [[sent], [entry[1][i]], [tag], entry[2]]

                if len(r1_results) % 1000 == 0:
                    print "finished {}".format(len(r1_results))

        print "Easy Set: ", count
        with open(output, 'w') as f:
            json.dump(r1_results, f, indent=2)  

        return r1_results 
    
    #results1 = get_func('../READY/r1_training_all.json', '../READY/r1_training_cleaned.json')    
    results2 = get_func('../READY/r2_training_all.json', '../READY/r2_training_cleaned.json')
    
    with open('../READY/r1_training_cleaned.json') as f:
        results1 = json.load(f)
    with open('../READY/r2_training_cleaned.json') as f:
        results2 = json.load(f)
    
    results2.update(results1)
    with open('../READY/full_cleaned.json', 'w') as f:
        json.dump(results2, f, indent=2)