import json

with open('../data/test_id.json') as f:
	whole_test = json.load(f)

with open('../READY/r1_training_cleaned.json') as f:
	r1_keys = json.load(f).keys()

with open('../READY/r2_training_cleaned.json') as f:
	r2_keys = json.load(f).keys()

r1 = []
r2 = []
for k in whole_test:
	if k in r1_keys:
		r1.append(k)
	elif k in r2_keys:
		r2.append(k)

with open('../data/simple_test_id.json', 'w') as f:
	json.dump(r1, f, indent=2)

with open('../data/complex_test_id.json', 'w') as f:
	json.dump(r2, f, indent=2)