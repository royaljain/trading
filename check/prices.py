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

def prices(ls_symbols):

    dt_start = dt.datetime(2006, 1, 1)
    dt_end = dt.datetime(2014, 12, 31)
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))

    dataobj = da.DataAccess('Yahoo')
    ls_symbols.append('SPY')

    ls_keys = ['close']
    ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    np.savetxt("/home/royal/projects/market/data/prices.csv", d_data['close'])

ls_symbols = ['AAPL', 'GOOG', 'MSFT'] 
prices(ls_symbols)
