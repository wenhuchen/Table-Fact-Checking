# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import json
import os
import re
from collections import Counter

#option = sys.argv[1]

folder = '../all_programs/'

failed = 0
success = 0
results = []

with open('../data/train_id.json') as f:
    train_id = json.load(f)
with open('../data/val_id.json') as f:
    dev_id = json.load(f)
with open('../data/test_id.json') as f:
    test_id = json.load(f)
with open('../data/complex_test_id.json') as f:
    complex_test_id = json.load(f)
with open('../data/simple_test_id.json') as f:
    simple_test_id = json.load(f)
with open('../data/small_test_id.json') as f:
    small_test_id = json.load(f)

word_counter = Counter()

fact_pattern = re.compile('#(.*?);[c,h]+-*[0-9]+#')
triggers = {}
words = ['not', 'no', 'never', "didn't", "won't", "wasn't", "isn't",
         "haven't", "weren't", "won't", 'neither', 'none', 'unable',
         'fail', 'different', 'outside', 'unable', 'fail', 'not_within_s_s',
         'not_within_n_n']
for w in words:
    triggers[w] = ['not_eq', 'not_str_eq', 'not_within_s_s', 'not_within_s_s',
                   'filter_not_eq', 'filter_str_not_eq', 'none',
                                    'all_str_not_eq', 'all_not_eq']

words = ['all', 'every', 'each']
for w in words:
    triggers[w] = ['all_eq', 'all_less', 'all_greater', 'all_greater',
                   'all_str_not_eq', 'all_not_eq', 'all_less_eq', 'all_greater_eq']

pos_triggers = {}
words = ['RBR', 'RBS', 'JJR', 'JJS']
for w in words:
    pos_triggers[w] = ['max', 'min', 'argmax', 'argmin', 'most_freq', 'filter_greater_eq',
                       'filter_less_eq', 'filter_greater', 'filter_less', 'less', 'greater',
                       'all_less', 'all_greater', 'all_less_eq', 'all_greater_eq']

fw_train = open('../preprocessed_data_program/train.tsv', 'w')
fw_dev = open('../preprocessed_data_program/dev.tsv', 'w')
fw_test = open('../preprocessed_data_program/test.tsv', 'w')
fw_small_test = open('../preprocessed_data_program/small_test.tsv', 'w')
fw_simple_test = open('../preprocessed_data_program/simple_test.tsv', 'w')
fw_complex_test = open('../preprocessed_data_program/complex_test.tsv', 'w')
pair_wise = []
for prog in os.listdir(folder):
    if prog.endswith('.json'):
        with open(folder + prog, 'r') as f:
            data = json.load(f)

        if len(data[4]) == 0:
            failed += 1
        else:
            success += 1

        if len(data[4]) > 50:
            data[4] = data[4][:50]
        else:
            data[4] = data[4]
        data.append(prog[:-5])
        results.append(data)
        """
		for r in data[4]:
			if r.endswith('True'):
				r = r[:-5]
				returned = True
				returned_lab  = "1"
			if r.endswith('False'):
				r = r[:-6]
				returned = False
				returned_lab = "0"
			r = r.replace(';', ' ;')
			r = r.replace('{', ' { ')
			r = r.replace('}', ' } ')
			text = ' '.join([x.strip() for x in re.split(fact_pattern, data[1]) if len(x) >0])
			word_counter.update(text.split())
			word_counter.update([_ for _ in r.split(' ') if len(_) > 0])
		"""
        if len(data[4]) == 0:
            for fw, ids in zip([fw_train, fw_dev, fw_test, fw_simple_test, fw_complex_test, fw_small_test], [train_id, dev_id, test_id, simple_test_id, complex_test_id, small_test_id]):
                if data[0] in ids:
                    text = ' '.join([x.strip() for x in re.split(fact_pattern, data[1]) if len(x.strip()) > 0])
                    if (returned and data[3] == 1) or ((not returned) and data[3] == 0):
                        fw.write(data[0] + "\t" + prog[:-5] + "\t" + str(data[3]) + "\t" +
                                 "0" + "\t" + text + "\t" + "no program" + "\t" + "1" + "\n")
                    else:
                        fw.write(data[0] + "\t" + prog[:-5] + "\t" + str(data[3]) + "\t" +
                                 "0" + "\t" + text + "\t" + "no program" + "\t" + "0" + "\n")
        else:
            for r in data[4]:
                if r.endswith('True'):
                    r = r[:-5]
                    returned = True
                    returned_lab = "1"
                    #r += " = True"
                if r.endswith('False'):
                    r = r[:-6]
                    returned = False
                    returned_lab = "0"
                    #r += " = False"
                r = r.replace(';', ' ;')
                r = r.replace('{', ' { ')
                r = r.replace('}', ' } ')
                for fw, ids in zip([fw_train, fw_dev, fw_test, fw_simple_test, fw_complex_test, fw_small_test], [train_id, dev_id, test_id, simple_test_id, complex_test_id, small_test_id]):
                    if data[0] in ids:
                        text = ' '.join([x.strip() for x in re.split(fact_pattern, data[1]) if len(x) > 0])
                        if (returned and data[3] == 1) or ((not returned) and data[3] == 0):
                            fw.write(data[0] + "\t" + prog[:-5] + "\t" + str(data[3]) + "\t" +
                                     returned_lab + "\t" + text + "\t" + r + "\t" + "1" + "\n")
                        else:
                            fw.write(data[0] + "\t" + prog[:-5] + "\t" + str(data[3]) + "\t" +
                                     returned_lab + "\t" + text + "\t" + r + "\t" + "0" + "\n")
                word_counter.update(text.split())
                word_counter.update([_ for _ in r.split(' ') if len(_) > 0])
fw_train.close()
fw_dev.close()
fw_test.close()
fw_simple_test.close()
fw_complex_test.close()
fw_small_test.close()
vocab = {"<PAD>": 0, "<UNK>": 1, "<SEP>": 2, "<CLS>": 3}
for k, v in word_counter.most_common():
    if v > 2:
        vocab[k] = len(vocab)

with open('../preprocessed_data_program/vocab.json', 'w') as f:
    json.dump(vocab, f, indent=2)

print("number of vocab: {}".format(len(vocab)))

with open('../preprocessed_data_program/all_programs.json', 'w') as f:
    json.dump(results, f, indent=2)

print("success: {}, failed: {}".format(success, failed))
