import json
from collections import Counter
import random

def get_instanaces(filename):
    with open(filename) as f:
        data = json.load(f)

    insts = []
    #pos_length = 0
    #neg_length = 0
    for k in data:
        text = data[k]['text']
        label = data[k]['label']
        for t, l in zip(text, label):
            if l == 1:
                insts.append((t, 1))
            else:
                insts.append((t, 0))
    
    print(len(insts))
    #print((pos_length + 0.0) / len(pos))
    #print((neg_length + 0.0) / len(neg))
    random.shuffle(insts)
    return insts

def get_vocab(instances):
    word_counter = Counter()
    for inst, t in instances:
        words = inst.split(' ')
        word_counter.update(words)
    words = word_counter.most_common()
    vocab = {"EOS": 0, "UNK": 1}
    for w, i in words:
        vocab[w] = len(vocab)
    return vocab
    
if __name__ == "__main__":
    instances = get_instanaces("../Interface/READY/prelim.json")
    test_split = instances[:2000]
    train_split = instances[2000:]
    vocab = get_vocab(train_split)
    
    #print(get_vocab(pos))
    #print(get_vocab(neg))