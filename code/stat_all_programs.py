import json
import os

failed = 0
success = 0
results = []
for prog in os.listdir('../data/all_program/'):
	if prog.endswith('.json'):
		with open('../data/all_program/' + prog, 'r') as f:
			data = json.load(f) 
		if len(data[3]) == 0:
			failed += 1
		else:
			success += 1

		results.append(data)

#with open('../READY/all_programs.json', 'w') as f:
#	json.dump(results, f)

print "success: {}, failed: {}".format(success, failed)
