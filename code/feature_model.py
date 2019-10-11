# encoding=utf8
import json
import os
import sys
import re
from collections import OrderedDict
reload(sys)
sys.setdefaultencoding('utf8')

with open('../READY/all_programs.json') as f:
    data = json.load(f)

with open('../data/small_test_id.json') as f:
    test_id = json.load(f)

features = {}
# features['more']
fact_pattern = re.compile('#(.*?);[c,h]+-*[0-9]+#')
#corr, fals = 0, 0
# for hyper in range(10):
tp, tn, fp, fn = 0, 0, 0, 0
orig_corr, orig_fals = 0, 0
pred_corr, pred_fals = 0, 0
results = {}
for line in data:
    if line[0] in test_id:
        gt = line[3]
        if gt:
            orig_corr += 1
        else:
            orig_fals += 1

        if len(line[4]) == 0:
            pred = 0
        else:
            preds = []
            for r in line[4]:
                if r.endswith('False'):
                    preds.append(0)
                else:
                    preds.append(1)
            if sum(preds) * 2 <= len(preds):
                pred = 0
            else:
                pred = 1

        if pred:
            pred_corr += 1
        else:
            pred_fals += 1

        text = ' '.join([x.strip() for x in re.split(fact_pattern, line[1]) if len(x.strip()) > 0])
        if pred == 1 and gt == 1:
            tp += 1
            if line[0] not in results:
                results[line[0]] = [OrderedDict({'fact': text, 'gold': line[3], 'pred': line[3]})]
            else:
                results[line[0]].append(OrderedDict({'fact': text, 'gold': line[3], 'pred': line[3]}))
        elif pred == 1 and gt == 0:
            fp += 1
            if line[0] not in results:
                results[line[0]] = [OrderedDict({'fact': text, 'gold': line[3], 'pred': 1 - line[3]})]
            else:
                results[line[0]].append(OrderedDict({'fact': text, 'gold': line[3], 'pred': 1 - line[3]}))
        elif pred == 0 and gt == 1:
            fn += 1
            if line[0] not in results:
                results[line[0]] = [OrderedDict({'fact': text, 'gold': line[3], 'pred': line[3]})]
            else:
                results[line[0]].append(OrderedDict({'fact': text, 'gold': line[3], 'pred': line[3]}))
        else:
            tn += 1
            if line[0] not in results:
                results[line[0]] = [OrderedDict({'fact': text, 'gold': line[3], 'pred': 1 - line[3]})]
            else:
                results[line[0]].append(OrderedDict({'fact': text, 'gold': line[3], 'pred': 1 - line[3]}))

print("original correct: {}, original false: {}".format(orig_corr, orig_fals))
print("pred correct: {}, pred false: {}".format(pred_corr, pred_fals))
print("TP: {}, FP: {}, FN: {}, TN: {}".format(tp, fp, fn, tn))
print("Accuracy: {}".format((tp + tn) / (tp + fp + fn + tn + 0.)))
with open('/tmp/eval_results.json', 'w') as f:
    json.dump(results, f, indent=2)
