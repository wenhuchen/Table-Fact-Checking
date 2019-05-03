# encoding=utf8
import sys
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pandas
import json
import re
import string
reload(sys)
sys.setdefaultencoding('utf8')

stop_words = ['be', 'she', 'he', 'her', 'his', 'their', 'the', 'it', ',', '.', '-', 'also', 'will', 'would', 'this', 'that',
             'these', 'those', 'well', 'with', 'on', 'at', 'and', 'as', 'for', 'from', 'in', 'its', 'of', 'to', 'a',
             'an', 'where', 'when', 'by', 'not', "'s", "'nt", "make", 'who', 'have', 'within', 'without', 'what',
             'during', 'than', 'then', 'if', 'when', 'while', 'time', 'appear', 'attend', 'every', 'one', 'two', 'over',
             'both', 'above', 'only']

#test = sys.argv[1]   
def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def get_closest(string, indexes, tab):
    dist = 10000
    len_string = len(string)
    for index in indexes:
        len_tab = len(tab[index[0]][index[1]])
        if abs(len_tab - len_string) == 0:
            return index
        elif abs(len_tab - len_string) < dist:
            minimum = index
            dist = abs(len_tab - len_string)
    return minimum

def replace_number(string):
    string = re.sub(r'(\b)one(\b)', r'\g<1>1\g<2>', string)
    string = re.sub(r'(\b)is one(\b)', r'\g<1>is 1\g<2>', string)
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

def postprocess(inp, backbone, tabs):
    new_str = []
    new_tags = []
    buf = ""
    last = set()
    inp, pos_tags = get_lemmatize(inp, True)
    for w, p in zip(inp, pos_tags):
        if w in backbone:
            if buf == "":
                last = set(backbone[w])
                buf = w
            else:
                proposed = set(backbone[w]) & last
                if not proposed:
                    if buf not in stop_words:
                        closest = get_closest(buf, last, tabs)
                        buf = '#' + buf + '#' + json.dumps(closest)
                    new_str.append(buf)
                    new_tags.append('ENT')
                    buf = w
                    last = set(backbone[w])
                else:
                    buf += " " + w
                    last = proposed
        else:
            if buf != "":
                if buf not in stop_words:
                    closest = get_closest(buf, last, tabs)
                    buf = '#' + buf + '#' + json.dumps(closest)
                new_str.append(buf)
                new_tags.append('ENT')
            buf = ""
            last = set()
            new_str.append(replace_number(w))
            new_tags.append(p)
    
    if buf != "":
        if buf not in stop_words:
            closest = get_closest(buf, last, tabs)
            buf = '#' + buf + '#' + json.dumps(closest)
        new_str.append(buf)
        new_tags.append("ENT")
    return " ".join(new_str), " ".join(new_tags)

def get_lemmatize(words, return_pos):
    #words = nltk.word_tokenize(words)
    words = words.strip().split(' ')
    try:
        pos_tags = [_[1] for _ in nltk.pos_tag(words)]
    except Exception:
        import pdb
        pdb.set_trace()
    word_roots = []
    for w, p in zip(words, pos_tags):
        if is_ascii(w) and p in tag_dict:
            word_roots.append(lemmatizer.lemmatize(w, tag_dict[p]))
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

with open('data/short_subset.txt') as f:
    limit_length = [_.strip() for _ in f.readlines()]

round1 = False
"""
if round1:
    t = pandas.read_csv('data/clean/positive.csv')
    t = t[t.AssignmentStatus=="Approved"]

    print len(t) * 10
    num = 10
    results = {}
    count = 0
    for i, row in t.iterrows():
        for j in range(1, num + 1):
            if row['Answer.A{}'.format(j)] == "Entailed":
                name = row['Input.url{}'.format(j)].split('/')[-1]
                if name in limit_length:
                    backbone = {}
                    tabs = []
                    with open('data/all_csv/' + name, 'r') as f:
                        for k, _ in enumerate(f.readlines()):
                            tabs.append([])
                            for l, w in enumerate(_.strip().split('#')):
                                tabs[-1].append(w)
                                w = get_lemmatize(w, False)
                                for sub in w:
                                    if sub not in backbone:
                                        backbone[sub] = [(k, l)]
                                    else:
                                        backbone[sub].append((k, l))
                    count += 1
                    if name in results:
                        sent, tag = postprocess(row['Input.s{}'.format(j)], backbone, tabs)
                        results[name][0].append(sent)
                        results[name][1].append(1)
                        results[name][2].append(tag)
                    else:
                        sent, tag = postprocess(row['Input.s{}'.format(j)], backbone, tabs)
                        results[name] = [[sent], [1], [tag]]
        if i // 100 == 1:
            print "finished {}/{}".format(i, len(t))
        #    break

    print count

    t = pandas.read_csv('data/clean/negative.csv')
    t = t[t.AssignmentStatus=="Approved"]

    print len(t) * 10
    count = 0
    for i, row in t.iterrows():
        for j in range(1, num + 1):
            if row['Answer.A{}'.format(j)] == "Refuted":
                name = row['Input.url{}'.format(j)].split('/')[-1]
                if name in limit_length:
                    backbone = {}
                    tabs = []
                    with open('data/all_csv/' + name, 'r') as f:
                        for k, _ in enumerate(f.readlines()):
                            tabs.append([])
                            for l, w in enumerate(_.strip().split('#')):
                                tabs[-1].append(w)
                                w = get_lemmatize(w, False)
                                for sub in w:
                                    if sub not in backbone:
                                        backbone[sub] = [(k, l)]
                                    else:
                                        backbone[sub].append((k, l))
                    count += 1
                    if name in results:
                        sent, tag = postprocess(row['Input.s{}'.format(j)], backbone, tabs)
                        results[name][0].append(sent)
                        results[name][1].append(0)
                        results[name][2].append(tag)
                    else:
                        sent, tag = postprocess(row['Input.s{}'.format(j)], backbone, tabs)
                        results[name] = [[sent], [0], [tag]]
        #if i // 100 == 1:
        #    break

    print count

    with open('READY/r1_training_cleaned.json', 'w') as f:
        json.dump(results, f, indent=2)
"""
with open('READY/r1_training_all.json') as f:
    data = json.load(f)
r1_results = {}
count = 0
for name in data:
    entry = data[name]
    for i in range(len(entry[0])):
        backbone = {}
        tabs = []
        with open('data/all_csv/' + name, 'r') as f:
            for k, _ in enumerate(f.readlines()):
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

        count += 1
        if name in r1_results:
            sent, tag = postprocess(entry[0][i], backbone, tabs)
            r1_results[name][0].append(sent)
            r1_results[name][1].append(entry[1][i])
            r1_results[name][2].append(tag)
        else:
            sent, tag = postprocess(entry[0][i], backbone, tabs)
            r1_results[name] = [[sent], [entry[1][i]], [tag]]

        if len(r1_results) % 1000 == 0:
            print "finished {}".format(len(r1_results))

    if len(r1_results) // 10 > 1:
        break
print "Easy Set: ", count

with open('READY/r1_training_cleaned.json', 'w') as f:
    json.dump(r1_results, f, indent=2)   

with open('READY/r2_training_all.json') as f:
    data = json.load(f)
r2_results = {}
count = 0
for name in data:
    #if name == "2-15295737-110.html.csv":
    #    import pdb
    #    pdb.set_trace()
    entry = data[name]
    for i in range(len(entry[0])):
        backbone = {}
        tabs = []
        with open('data/all_csv/' + name, 'r') as f:
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

        count += 1
        if "paula" in entry[0][i]:
            import pdb
            pdb.set_trace()
        if name in r2_results:
            sent, tag = postprocess(entry[0][i], backbone, tabs)
            r2_results[name][0].append(sent)
            r2_results[name][1].append(entry[1][i])
            r2_results[name][2].append(tag)
        else:
            sent, tag = postprocess(entry[0][i], backbone, tabs)
            r2_results[name] = [[sent], [entry[1][i]], [tag]]
        
        if len(r2_results) % 1000 == 0:
            print "finished {}".format(len(r2_results))

    if len(r2_results) // 10 > 1:
        break
print "Hard Set: ", count

with open('READY/r2_training_cleaned.json', 'w') as f:
    json.dump(r2_results, f, indent=2)

r2_results.update(r1_results)
with open('READY/full_cleaned.json', 'w') as f:
    json.dump(r2_results, f, indent=2)

