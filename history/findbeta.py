
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

def findbeta(ls_symbols):
    dt_start = dt.datetime(2006, 1, 1)
    dt_end = dt.datetime(2014, 12, 31)
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))

    dataobj = da.DataAccess('Yahoo')
    ls_symbols.append('SPY')

    ls_keys = ['close']
    ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))
    betas = []
    
    for i in range(0, len(ls_symbols)):
        regr = linear_model.LinearRegression()
        
        x = d_data['close'].ix[:,i]
        y = d_data['close'].ix[:, (len(ls_symbols)-1)]
        
        xf = []
        yf = []
        for i in range(0,len(x)):
            if not np.isnan(x[i]) and not np.isnan(y[i]):
                xf.append(x[i])
                yf.append(y[i])
        
        xff = []
        yff = []
        
        for i in range(0,len(xf)-1):
            xff.append(xf[i+1]-xf[i])
            yff.append(yf[i+1]-yf[i])
        
        
        xff =np.asarray(xff)
        yff =np.asarray(yff)
        xff = xff.reshape((xff.shape[0],-1))
        yff =yff.reshape((yff.shape[0],-1))
        regr.fit(yff, xff)
        betas.append(regr.coef_)

    return betas

betas = findbeta(['AAPL', 'GOOG', 'MSFT'])
