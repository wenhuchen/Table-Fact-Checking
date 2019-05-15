import json
import os
import re
from collections import Counter

folder = '../data/all_programs/'

failed = 0
success = 0
results = []

word_counter = Counter()
"""
for idx in range(0, 118389):
	if not os.path.exists('../data/all_programs/nt-{}.json'.format(idx)):
		print "nt-{}".format(idx)
"""
#with open('../READY/preprocessed_0.json'):

for prog in os.listdir('../data/all_programs/'):
	if prog.endswith('.json'):
		with open(folder + prog, 'r') as f:
			data = json.load(f) 
		
		if len(data[4]) == 0:
			failed += 1
		else:
			success += 1
		
		word_counter.update(data[2].split(' '))
		results.append(data)

vocab = {"<PAD>": 0, "<UNK>": 1, "<SEP>": 2, "<CLS>": 3}
for k, v in word_counter.most_common():
	if v > 2:
		vocab[k] = len(vocab)

with open('../data/code_vocab.txt') as f:
    words = [_.strip() for _ in f.readlines()]
    API_vocab = {"<PAD>": 0, "<CLS>": 1}
    for i, w in enumerate(words):
    	API_vocab[w] = len(API_vocab)

with open('../data/vocab.json', 'w') as f:
	json.dump({"s_vocab": vocab, "a_vocab": API_vocab}, f, indent=2)

print "number of vocab: {}".format(len(vocab))

with open('../READY/all_programs.json', 'w') as f:
	json.dump(results, f, indent=2)


print "success: {}, failed: {}".format(success, failed)
