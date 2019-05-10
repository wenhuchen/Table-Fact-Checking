import json
import os

folder = '../data/all_programs/'

failed = 0
success = 0
results = []

"""
for idx in range(0, 118389):
	if not os.path.exists('../data/all_programs/nt-{}.json'.format(idx)):
		print "nt-{}".format(idx)
"""
for prog in os.listdir('../data/all_programs/'):
	if prog.endswith('.json'):
		with open(folder + prog, 'r') as f:
			data = json.load(f) 
		
		if len(data[3]) == 0:
			failed += 1
		else:
			success += 1

		if 'third' in data[1]:
			new_line = []
			for d in data[3]:
				new_line.append(d.replace(r'second(', 'third('))
			data[3] = new_line

		results.append(data)

#files = []
with open('../READY/all_programs.json', 'w') as f:
	json.dump(results, f, indent=2)


#print "success: {}, failed: {}".format(success, failed)
