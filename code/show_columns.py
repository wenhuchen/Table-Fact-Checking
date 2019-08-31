import os
import pandas
import re
import json
import shutil

"""
items = os.listdir("../data/all_csv/")

pattern_left = r'[0-9]+(?= \-)'
pattern_right = r'(?<=\- )[0-9]+'
#newlist = []
for name in items:
	name = "../data/all_csv/{}".format(name)
	if name.endswith('csv'):
		data = pandas.read_csv(name, '#')
		row = data.iloc[1]
		for k in row.keys():
			if isinstance(row[k], str) and re.match(r'[0-9]+ \- [0-9]+', row[k]):
				if "score" in k:
					candidates_left = []
					candidates_right = []
					for i in range(len(data)):
						groups = re.search(pattern_left, data.iloc[i][k]):
						candidates_left.append(re.search(pattern_left, data.iloc[i][k]).group())						
						candidates_right.append(re.search(pattern_right, data.iloc[i][k]).group())
					data.rename(columns = {k: 'aggregate ' + k}, inplace = True)
					data[k + " (left)"] = candidates_left
					data[k + " (right)"] = candidates_right
					print data
				elif k == "result":
					candidates_left = []
					candidates_right = []
					for i in range(len(data)):
						candidates_left.append(re.search(pattern_left, data.iloc[i][k]).group())						
						candidates_right.append(re.search(pattern_right, data.iloc[i][k]).group())
					data.rename(columns = {'result': 'aggregate result'}, inplace = True)
					data[k + " (left)"] = candidates_left
					data[k + " (right)"] = candidates_right
					print data
				elif k in ["height", "record"]:
					continue
				else:
					print k
					print row[k]
					print

with open('../data/complex_ids.json') as f:
	complex_id = set(json.load(f))

with open('/tmp/clean_files.txt') as f:
	files = map(lambda x:x.strip(), f.readlines())

cleaned_complex = set(files) & set(complex_id)

with open('../data/cleaned_complex_ids.json', 'w') as f:
	json.dump(list(cleaned_complex), f)
"""
with open('../READY/r1_training_all.json') as f:
	files_1 = json.load(f)
with open('../READY/r2_training_all.json') as f:
	files_2 = json.load(f)
files_1.update(files_2)

files = set(files_1)

items = os.listdir("../data/all_csv/")
for name in items:
	if name.endswith('csv'):
		if name not in files:
			full_name = "../data/all_csv/{}".format(name)
			trash_name = "../data/trash_csv/{}".format(name)
			shutil.move(full_name, trash_name)
