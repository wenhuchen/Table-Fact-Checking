# encoding=utf8
import json
import pandas
import numpy
from beam_search import dynamic_programming
#from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool
import multiprocessing
import sys
reload(sys)
sys.setdefaultencoding('utf8')

with open('../READY/full_cleaned.json') as f:
	data = json.load(f)

def isnumber(string):
	return string in [numpy.dtype('int64'), numpy.dtype('int32'), numpy.dtype('float32'), numpy.dtype('float64')]

def replace(string):
	"""
	string = string.replace(' ', '_')
	string = string.replace('/', 'or')
	string = string.replace('(', '')
	string = string.replace(')', '')
	string = string.replace('=', '_eq_')
	string = string.replace('<', '_lt_')
	string = string.replace('>', '_gt_')
	string = string.replace('class', 'cls')
	string = string.replace('-', 'minus')
	string = string.replace('+', 'plus')
	string = string.replace(',', '')
	string = string.replace('.', '')
	string = string.replace("'", '')
	string = string.replace("%", 'percent')
	string = string.replace(":", '_')
	if string[0].isdigit():
		string = "_" + string
	"""
	return string

def list2tuple(inputs):
    mem = []
    for s in inputs:
        mem.append(tuple(s))
    return mem

debug = False
if debug:
	count = 0
	preprocessed = []
	for table_name in data:
		t = pandas.read_csv('../data/all_csv/{}'.format(table_name), delimiter="#")
		cols = t.columns
		cols = cols.map(lambda x: replace(x) if isinstance(x, (str, unicode)) else x)
		t.columns = cols
		mapping = {i: "num" if isnumber(t) else "str" for i, t in enumerate(t.dtypes)}
		entry = data[table_name]
		for sent, pos_tag in zip(entry[0], entry[2]):
			count += 1
			inside = False
			position = False
			masked_sent = ''
			position_buf, mention_buf = '', ''
			mem_num, head_num, mem_str, head_str = [], [], [], []
			for n in range(len(sent)):
				if sent[n] == '#':
					if position:
						pos = json.loads(position_buf)
						if mapping[pos[1]] == 'num':
							if pos[0] == 0:
								if cols[pos[1]] not in head_num:
									head_num.append(cols[pos[1]])
							else:
								mem_num.append((cols[pos[1]], numpy.asscalar(t.at[pos[0] - 1, cols[pos[1]]])))
						else:
							if pos[0] == 0:
								if cols[pos[1]] not in head_str:
									head_str.append(cols[pos[1]])
							else:
								mem_str.append((cols[pos[1]], t.at[pos[0] - 1, cols[pos[1]]]))
						masked_sent += "<ENTITY>"
						position_buf = ""
						mention_buf = ""
						inside = False
						position = False
					else:
						inside = True
				elif sent[n] == ';':
					position = True
				else:
					if position:
						position_buf += sent[n]
					elif inside:
						mention_buf += sent[n]
					else:
						masked_sent += sent[n]
			for k, v in mem_num:
				if k not in head_num:
					head_num.append(k)
			for k, v in mem_str:
				if k not in head_str:
					head_str.append(k)
			
			#preprocessed.append((k, sent, masked_sent, pos, mem_num, head_num, ))
			#print sent
			#print masked_sent
			#print mem_num, head_num, mem_str, head_str, pos_tag
			#print
			preprocessed.append((table_name, sent, pos_tag, masked_sent, mem_str, mem_num, head_str, head_num))
			#dynamic_programming(table_name, t, sent, masked_sent, pos_tag, mem_str, mem_num, head_str, head_num, 2)
	with open('../READY/preprocessed.json', 'w') as f:
		json.dump(preprocessed, f, indent=2)

else:
	with open('../READY/preprocessed.json', 'r') as f:
		data = json.load(f)

	def func(args):
		table_name, sent, pos_tag, masked_sent, mem_str, mem_num, head_str, head_num = args
		t = pandas.read_csv('../data/all_csv/{}'.format(table_name), delimiter="#")
		#print table_name, sent, pos_tag, masked_sent, mem_str, mem_num, head_str, head_num
		cols = t.columns
		cols = cols.map(lambda x: replace(x) if isinstance(x, (str, unicode)) else x)
		t.columns = cols
		dynamic_programming(table_name, t, sent, masked_sent, pos_tag, mem_str, mem_num, head_str, head_num)

	data = data[:24]
	table_name = [_[0] for _ in data]
	sent = [_[1] for _ in data]
	pos_tag = [_[2] for _ in data]
	masked_sent = [_[3] for _ in data]
	mem_str = [list2tuple(_[4]) for _ in data]
	mem_num = [list2tuple(_[5]) for _ in data]
	head_str = [_[6] for _ in data]
	head_num = [_[7] for _ in data]
	
	cores = multiprocessing.cpu_count()
	print "Using {} cores".format(cores)
	pool = Pool(cores)

	res = pool.map(func, zip(table_name, sent, pos_tag, masked_sent, mem_str, mem_num, head_str, head_num))
	#print res
	
	pool.close()
	pool.join()
	"""
	for arg in zip(table_name, sent, pos_tag, masked_sent, mem_str, mem_num, head_str, head_num):
		func(arg)
		print "finished"
	"""
