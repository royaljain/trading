import pandas as pd
import numpy as np
import math
import copy
import QSTK.qstkutil.qsdateutil as du
import datetime as dt
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkstudy.EventProfiler as ep
from sklearn import datasets, linear_model



def expectation(ls_symbols):

	dt_start = dt.datetime(2006, 1, 1)
	dt_end = dt.datetime(2014, 12, 31)
	ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))

	dataobj = da.DataAccess('Yahoo')

	ls_keys = ['close']
	ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
	d_data = dict(zip(ls_keys, ldf_data))


	n = len(ls_symbols)

	exp =  np.zeros(n)

	for i in range(0, n):

		x = d_data['close'].ix[:,i]
		
		xf = []
		for k in range(0,len(x)):
			if not np.isnan(x[k]) :
				xf.append(x[k])
				
		
		exp[i] = np.mean(xf)
	
	return exp

