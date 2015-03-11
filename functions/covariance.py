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



def covariance(ls_symbols):

	dt_start = dt.datetime(2006, 1, 1)
	dt_end = dt.datetime(2014, 12, 31)
	ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))

	dataobj = da.DataAccess('Yahoo')

	ls_keys = ['close']
	ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
	d_data = dict(zip(ls_keys, ldf_data))


	n = len(ls_symbols)

	covar =  np.zeros((n, n))

	for i in range(0, n):
		for j in range(0, n):
			
			x = d_data['close'].ix[:,i]
			y = d_data['close'].ix[:,j]
			
			xf = []
			yf = []
			for k in range(0,len(x)):
				if not np.isnan(x[k]) and not np.isnan(y[k]):
					xf.append(x[k])
					yf.append(y[k])
			
			m = np.zeros((2, len(xf)))
			m[0, :] = xf
			m[1, :] = yf
			a = np.cov(m)
			
			covar[i][j] = a[0][1]
	
	return covar

