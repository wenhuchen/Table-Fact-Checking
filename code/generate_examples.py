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
from multiprocessing import Pool
import multiprocessing

reload(sys)
sys.setdefaultencoding('utf8')

#stop_words = ['be', 'she', 'he', 'her', 'his', 'their', 'the', 'it', ',', '.', '-', 'also', 'will', 'would', 'this', 'that',
#             'these', 'those', 'well', 'with', 'on', 'at', 'and', 'as', 'for', 'from', 'in', 'its', 'of', 'to', 'a',
#             'an', 'where', 'when', 'by', 'not', "'s", "'nt", "make", 'who', 'have', 'within', 'without', 'what',
#             'during', 'than', 'then', 'if', 'when', 'while', 'time', 'appear', 'attend', 'every', 'one', 'two', 'over',
#            'both', 'above', 'only', ",", ".", "(", ")", "&", ":"]
with open('../data/vocab.json') as f:
    vocab = json.load(f)

#useless_words = [',', '.', "'s"]
with open('../data/stop_words.json') as f:
    stop_words = json.load(f)

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def augment(s):
    if 'first' in s:
        s.append("1st")
    elif 'second' in s:
        s.append("2nd")
    elif 'third' in s:
        s.append("3rd")
    elif 'fourth' in s:
        s.append("4th")
    elif 'fifth' in s:
        s.append("5th")
    elif 'sixth' in s:
        s.append("6th")
    elif '01' in s:
        s.append("1")
    elif '02' in s:
        s.append("2")
    elif '03' in s:
        s.append("3")
    elif '04' in s:
        s.append("4")
    elif '05' in s:
        s.append("5")
    elif '06' in s:
        s.append("6")
    elif '07' in s:
        s.append("7")
    elif '08' in s:
        s.append("8")
    elif '09' in s:
        s.append("9") 
    elif 'crowd' in s:
        s.append("people")
    return s

def replace_useless(s):
    s = s.replace(',', '')
    s = s.replace('.', '')
    #s = s.replace('/', '')
    return s

def get_closest(inp, string, indexes, tab, threshold):
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

    vocabs = []
    for s in string.split(' '):
        vocabs.append(vocab.get(s, 2000))

    # String Length
    feature = [len_string]
    # Proportion
    feature.append(-dist / (len_string + dist + 0.) * 3)
    # Whether contain rare words
    if max(vocabs) > 600:
        feature.append(2)
    else:
        feature.append(-2)
    if max(vocabs) > 10000:
        feature.append(1000)
    else:
        feature.append(0)
    # Whether it is only a word
    if len_string > 1:
        feature.append(1)
    else:
        feature.append(0)
    # Whether candidate has only one
    if len(indexes) == 1:
        feature.append(1)
    else:
        feature.append(0)
    # Whether cover over half of it
    if len_string > dist:
        feature.append(1)
    else:
        feature.append(0)
    # Whether contains alternative
    cand = tab[minimum[0]][minimum[1]]
    if '(' in cand and ')' in cand:
        feature.append(2)
    else:
        feature.append(0)
    # Whether it is in order
    if string not in cand:
        feature.append(-2)
    else:
        feature.append(0)

    # Whether it is a header
    if minimum[0] == 0:
        feature.append(-1)
    else:
        feature.append(0)

    if string in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 
                  'september', 'october', 'november', 'december', 'jan', 'feb', 'mar', 
                  'apr', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec']:
        feature.append(1000)
    else:
        feature.append(0)

    #print string, "@", cand, "@", feature, "@", sum(feature) > 1.0, "@", " ".join(inp)
    if sum(feature) > threshold:
        return minimum
    else:
        return None
    """
    if dist < len_string:
        return minimum
    else:
        if dist > len_string * 2 or len(indexes) >= 3:
            return None
        else:
            use_minimum = False
            for s in string.split(' '):
                if vocab.get(s, 2000) >= 2000:
                    use_minimum = True
                    break
                else:
                    use_minimum = False

            if use_minimum:
                return minimum
            else:
                return None
    """

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

def postprocess(inp, backbone, trans_backbone, transliterate, tabs, recover_dict, threshold=1.0):
    new_str = []
    new_tags = []
    buf = ""
    pos_buf = []
    last = set()
    prev_closest = []
    inp, pos_tags = get_lemmatize(inp, True)
    for w, p in zip(inp, pos_tags):
        if w in backbone:
            if buf == "":
                last = set(backbone[w])
                buf = recover_dict.get(w, w)
                pos_buf.append(p)
            else:
                proposed = set(backbone[w]) & last
                if not proposed:
                    if buf not in stop_words:
                        closest = get_closest(inp, buf, last, tabs, threshold)
                        if closest:
                            if closest[0] == 0:
                                buf = '#{};h{},{}#'.format(buf, closest[0], closest[1])
                            else:
                                buf = '#{};c{},{}#'.format(buf.title(), closest[0], closest[1])
                    new_str.append(buf)
                    if buf.startswith("#"):
                        new_tags.append('ENT')
                    else:
                        new_tags.extend(pos_buf)
                    pos_buf = []
                    buf = recover_dict.get(w, w)
                    last = set(backbone[w])
                    pos_buf.append(p)
                else:
                    last = proposed
                    buf += " " + recover_dict.get(w, w)
                    pos_buf.append(p)

        elif w in trans_backbone:
            if buf == "":
                last = set(trans_backbone[w])
                buf = transliterate[w]
                pos_buf.append(p)
            else:
                proposed = set(trans_backbone[w]) & last
                if not proposed:
                    if buf not in stop_words:
                        closest = get_closest(inp, buf, last, tabs, threshold)
                        if closest:
                            if closest[0] == 0:
                                buf = '#{};h{},{}#'.format(buf, closest[0], closest[1])
                            else:
                                buf = '#{};c{},{}#'.format(buf.title(), closest[0], closest[1])
                    new_str.append(buf)
                    if buf.startswith("#"):
                        new_tags.append('ENT')
                    else:
                        new_tags.extend(pos_buf)
                    pos_buf = []
                    buf = transliterate[w]
                    last = set(trans_backbone[w])
                    pos_buf.append(p)
                else:
                    buf += " " + transliterate[w]
                    last = proposed
                    pos_buf.append(p)
        
        else:
            if buf != "":
                if buf not in stop_words:
                    closest = get_closest(inp, buf, last, tabs, threshold)
                    if closest:
                        if closest[0] == 0:
                            buf = '#{};h{},{}#'.format(buf, closest[0], closest[1])
                        else:
                            buf = '#{};c{},{}#'.format(buf.title(), closest[0], closest[1])
                new_str.append(buf)
                if buf.startswith("#"):
                    new_tags.append('ENT')
                else:
                    new_tags.extend(pos_buf)
                pos_buf = []
            buf = ""
            last = set()
            new_str.append(replace_number(w))
            new_tags.append(p)
    
    if buf != "":
        if buf not in stop_words:
            closest = get_closest(inp, buf, last, tabs, threshold)
            if closest:
                if closest[0] == 0:
                    buf = '#{};h{},{}#'.format(buf, closest[0], closest[1])
                else:
                    buf = '#{};c{},{}#'.format(buf.title(), closest[0], closest[1])
        new_str.append(buf)
        if buf.startswith("#"):
            new_tags.append('ENT')
        else:
            new_tags.extend(pos_buf)
        pos_buf = []
    
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

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def merge_strings(string, tags=None):
    buff = ""
    inside = False
    words = []

    for c in string:
        if c == "#" and not inside:
            inside = True
            buff += c
        elif c == "#" and inside:
            inside = False
            buff += c
            words.append(buff)
            buff = ""
        elif c == " " and not inside:
            if buff:
                words.append(buff)
            buff = ""
        elif c == " " and inside:
            buff += c
        else:
            buff += c

    if buff:
        words.append(buff)

    tags = tags.split(' ')
    assert len(words) == len(tags), "{} and {}".format(words, tags)
    i = 0
    while i < len(words):
        if i < 2:
            i += 1
        elif words[i].startswith('#') and (not words[i-1].startswith('#')) and words[i-2].startswith('#'):
            if is_number(words[i].split(';')[0][1:]) and is_number(words[i-2].split(';')[0][1:]):
                i += 1
            elif words[i].split(';')[1][:-1] == words[i-2].split(';')[1][:-1]:
                position = words[i].split(';')[1][:-1]
                candidate = words[i - 2].split(';')[0] + " " + words[i-1] + " " + words[i].split(';')[0][1:] + ";" + position + "#"
                words[i] = candidate
                del words[i-1]
                del tags[i-1]
                i -= 1
                del words[i-1]
                del tags[i-1]
                i -= 1
            else:
                i += 1
        else:
            i += 1
    
    return " ".join(words), " ".join(tags)

def sub_func(inputs):
    name, entry = inputs
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
                #w = w.replace(',', '').replace('  ', ' ')
                tabs[-1].append(w)
                if len(w) > 0:
                    lemmatized_w = get_lemmatize(w, False, recover_dict)
                    lemmatized_w = augment(lemmatized_w)
                    for sub in lemmatized_w:
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
                    for sub in w.split(' '):
                        if sub not in backbone:
                            backbone[sub] = [(k, l)]
                            if not is_ascii(sub):
                                trans_backbone[unidecode(sub)] = [(k, l)]
                                transliterate[unidecode(sub)] = sub
                        else:
                            if (k, l) not in backbone[sub]:
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
        sent, tags = postprocess(entry[0][i], backbone, trans_backbone, transliterate, tabs, recover_dict, threshold=1.0)
        if "#" not in sent:
            sent, tags = postprocess(entry[0][i], backbone, trans_backbone, transliterate, tabs, recover_dict, threshold=0.0)

        sent, tags = merge_strings(sent, tags)
        if i > 0:
            results[0].append(sent)
            results[1].append(entry[1][i])
            results[2].append(tags)
        else:
            results = [[sent], [entry[1][i]], [tags], entry[2]]

    return name, results

def get_func(filename, output):
    with open(filename) as f:
        data = json.load(f)
    r1_results = {}
    names = []
    entries = []
    for name in data:
        names.append(name)
        entries.append(data[name])
    
    cores = multiprocessing.cpu_count() - 2
    pool = Pool(cores)

    r = pool.map(sub_func, zip(names, entries))
    #for i in range(100):
    #    r = [sub_func((names[i], entries[i]))]

    pool.close()
    pool.join()
    
    return dict(r) 

results1 = get_func('../READY/r1_training_all.json', '../READY/r1_training_cleaned.json')    
results2 = get_func('../READY/r2_training_all.json', '../READY/r2_training_cleaned.json')

results2.update(results1)
with open('../READY/full_cleaned.json', 'w') as f:
    json.dump(results2, f, indent=2)