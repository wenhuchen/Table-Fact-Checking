# encoding=utf8
import json
import pandas
import numpy
from beam_search import dynamic_programming
from multiprocessing import Pool
import multiprocessing
import sys
import time
import argparse
import os
reload(sys)
sys.setdefaultencoding('utf8')

parser = argparse.ArgumentParser()
parser.add_argument("--synthesize", default=False, action="store_true", help="whether to synthesize data")
parser.add_argument("--sequential", default=False, action="store_true", help="Whether to use sequential or distributed")
parser.add_argument("--debug", default=False, action="store_true", help="Whether to use debugging mode")
parser.add_argument("--part", type=int, default=0, help="choose a part")
parser.add_argument("--split", type=int, default=1, help="how many splits")
args = parser.parse_args()

with open('../READY/full_cleaned.json') as f:
	data = json.load(f)

stop_words = ['be', 'she', 'he', 'her', 'his', 'their', 'the', 'it', ',', '.', '-', 'also', 'will', 'would', 'this', 'that',
             'these', 'those', 'well', 'with', 'on', 'at', 'and', 'as', 'for', 'from', 'in', 'its', 'of', 'to', 'a',
             'an', 'where', 'when', 'by', 'not', "'s", "'nt", "make", 'who', 'have', 'within', 'without', 'what',
             'during', 'than', 'then', 'if', 'when', 'while', 'time', 'appear', 'attend', 'every', 'one', 'two', 'over',
             'both', 'above', 'only']

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

if not args.synthesize:
	count = 0
	preprocessed = []
	for table_name in data:
		t = pandas.read_csv('../data/all_csv/{}'.format(table_name), delimiter="#")
		cols = t.columns
		#cols = cols.map(lambda x: replace(x) if isinstance(x, (str, unicode)) else x)
		t.columns = cols
		mapping = {i: "num" if isnumber(t) else "str" for i, t in enumerate(t.dtypes)}
		entry = data[table_name]
		caption = entry[3].split(' ')
		
		for sent, label, pos_tag in zip(entry[0], entry[1], entry[2]):
			count += 1
			inside = False
			position = False
			masked_sent = ''
			position_buf, mention_buf = '', ''
			mem_num, head_num, mem_str, head_str = [], [], [], []
			for n in range(len(sent)):
				if sent[n] == '#':
					if position:
						if position_buf.startswith('h'):
							idx = int(position_buf[1:])
							if mapping[idx] == 'num':
								if cols[idx] not in head_num:
									head_num.append(cols[idx])
							else:
								if cols[idx] not in head_str:
									head_str.append(cols[idx])
						if position_buf.startswith('c'):
							idx = int(position_buf[1:])
							if idx == -1:
								pass
							else:
								if mapping[idx] == 'num':
									if mention_buf.isdigit():
										mention_buf = int(mention_buf)
									else:
										mention_buf = float(mention_buf)
									val = (cols[idx], mention_buf)
									if val not in mem_num:
										mem_num.append(val)
								else:
									val = (cols[idx], mention_buf)
									if val not in mem_str:
										mem_str.append(val)
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

			tokens = masked_sent.split()
			i = 0
			while i < len(tokens):
				_ = tokens[i]
				if i + 1 < len(tokens):
					if _.isdigit() and (tokens[i + 1] not in ["thousand", "hundred"]):
						num = int(_)
						i += 1
					elif _.isdigit() and tokens[i + 1] in ["thousand", "hundred"]:
						if tokens[i + 1] == "thousand":
							num = int(_) * 1000
							i += 2
						elif tokens[i + 1] == "hundred":
							num = int(_) * 100
							i += 2
					elif _ == "a" and tokens[i + 1] in ["thousand", "hundred"]:
						if tokens[i + 1] == "thousand":
							num = 1000
							i += 2
						elif tokens[i + 1] == "hundred":
							num = 100
							i += 2
					elif '.' in tokens[i]:
						try:
							num = float(_)
							i += 1
						except Exception:
							i += 1
							continue
					else:
						i += 1
						continue
				else:
					if _.isdigit():
						num = int(_)
						i += 1
					elif '.' in tokens[i]:
						try:
							num = float(_)
							i += 1
						except Exception:
							i += 1
							continue
					else:
						i += 1
						continue
				
				flag = False				
				if i > 3 and i < len(tokens):
					if tokens[i - 2] == "than" or tokens[i - 3] == "than":
						for h in head_num:
							if num >= t[h].min() and num <= t[h].max():
								mem_num.append((h, num))
								print mem_num[-1], sent
								del head_num[head_num.index(h)]
								flag = True
								break
				
				if not flag:
					mem_num.append(("tmp_input", num))
				"""
				if num < len(t) or len(head_num) == 0 or ("JJR" not in pos_tag and "RBR" not in pos_tag):
					mem_num.append(("tmp_input", num))
				else:
					
				"""
			for k, v in mem_num:
				if k not in head_num and k != "tmp_input":
					head_num.append(k)
			for k, v in mem_str:
				if k not in head_str:
					head_str.append(k)

			#preprocessed.append((k, sent, masked_sent, pos, mem_num, head_num, ))
			#print sent
			#print masked_sent
			#print mem_num, head_num, mem_str, head_str, pos_tag
			preprocessed.append((table_name, sent, pos_tag, masked_sent, mem_str, 
			                     mem_num, head_str, head_num, "nt-{}".format(len(preprocessed)), label))
			#dynamic_programming(table_name, t, sent, masked_sent, pos_tag, mem_str, mem_num, head_str, head_num, 2)
	length = len(preprocessed) // args.split
	for i in range(args.split):
		with open('../READY/preprocessed_{}.json'.format(i), 'w') as f:
			if i == args.split - 1:
				json.dump(preprocessed[i * length :], f, indent=2)
			else:
				json.dump(preprocessed[i * length : (i+1) * length], f, indent=2)

else:
	with open('../READY/preprocessed_{}.json'.format(args.part), 'r') as f:
		data = json.load(f)

	def func(inputs):
		table_name, sent, pos_tag, masked_sent, mem_str, mem_num, head_str, head_num, idx, labels = inputs
		t = pandas.read_csv('../data/all_csv/{}'.format(table_name), delimiter="#", encoding = 'utf-8')
		t.fillna('')
		cols = t.columns
		cols = cols.map(lambda x: replace(x) if isinstance(x, (str, unicode)) else x)
		t.columns = cols
		if args.sequential:
			res = dynamic_programming(table_name, t, sent, masked_sent, pos_tag, mem_str, mem_num, head_str, head_num, labels, 7)
			print idx, res[:-1]
			for r in res[-1]:
				print r
		else:
			if not os.path.exists('../data/all_programs/'):
				os.mkdir('../data/all_programs/')
			try:
			    res = dynamic_programming(table_name, t, sent, masked_sent, pos_tag, mem_str, mem_num, head_str, head_num, labels, 7)
			    with open('../data/all_programs/{}.json'.format(idx), 'w') as f:
			        json.dump(res, f, indent=2)
			except Exception:
			    print "failed {}, {}".format(table_name, idx)

	table_name = [_[0] for _ in data]
	sent = [_[1] for _ in data]
	pos_tag = [_[2] for _ in data]
	masked_sent = [_[3] for _ in data]
	mem_str = [list2tuple(_[4]) for _ in data]
	mem_num = [list2tuple(_[5]) for _ in data]
	head_str = [_[6] for _ in data]
	head_num = [_[7] for _ in data]
	idxes = [_[8] for _ in data]
	labels = [_[9] for _ in data]	
	#with open('../data/unfinished.txt') as f:
	#	files = [_.strip() for _ in f.readlines()]
	if args.sequential:
		for arg in zip(table_name, sent, pos_tag, masked_sent, mem_str, mem_num, head_str, head_num, idxes, labels):
			if arg[8] in ["nt-65"]:
				func(arg)
	else:
		cores = multiprocessing.cpu_count() - 2
		print "Using {} cores".format(cores)
		pool = Pool(cores)
		res = pool.map(func, zip(table_name, sent, pos_tag, masked_sent, mem_str, mem_num, head_str, head_num, idxes, labels))
		
		pool.close()
		pool.join()