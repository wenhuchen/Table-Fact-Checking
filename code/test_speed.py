import pandas
import time
t = pandas.read_csv('../data/all_csv/2-10808089-16.html.csv', delimiter="#")

start_time = time.time()
for i in range(1000):
	t.iloc[0][0]
print "used {}".format(time.time() - start_time