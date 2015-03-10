import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.optimize import minimize, rosen, rosen_der


ls_symbols = ["AAPL", "GLD", "GOOG","XOM"]
dt_start = dt.datetime(2006, 1, 1)
dt_end = dt.datetime(2010, 12, 31)
dt_timeofday = dt.timedelta(hours=16)
ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

c_dataobj = da.DataAccess('Yahoo')
ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
d_data = dict(zip(ls_keys, ldf_data))

na_price = d_data['close'].values
na_normalized_price = na_price / na_price[0, :]


plt.clf()

plt.plot(ldt_timestamps, na_normalized_price)
plt.legend(ls_symbols)
plt.ylabel('Adjusted Close')
plt.xlabel('Date')
plt.savefig('before.pdf', format='pdf')

n = len(ls_symbols)
e=[]
f=[]

for i in range(0,n):
    e.append(d_data['close'][ls_symbols[i]].mean())
    f.append(d_data['close'][ls_symbols[i]].values)


m = np.zeros(shape=(n,n))

for i in range(0,n):
    m1 = np.mean(f[i])
    f[i][:] = [x - m1 for x in f[i]]

for i in range(0,n):
    for j in range(0,n):
        temp = [a*b for a,b in zip(f[i],f[j])]
        m[i][j] = np.mean(temp)



fun = lambda x: -(sum([i*j for (i, j) in zip(e,x)])/(((np.mat(x)*m)*(np.mat(x).T))[0,0])**0.5)


cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1 })
bnds = [(0.0,1.0)]
for i in range(0,n-1):
    bnds.append((0.0,1.0))

l = []
l.append(1)

for i in range(0,n-1):
    l.append(0)
	
res = minimize(fun,l, method='SLSQP', bounds=bnds,constraints=cons)


ls_symbols = ["AAPL", "GLD", "GOOG","XOM"]
dt_start = dt.datetime(2011, 1, 1)
dt_end = dt.datetime(2011, 1, 31)
dt_timeofday = dt.timedelta(hours=16)
ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

c_dataobj = da.DataAccess('Yahoo')
ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
d_data = dict(zip(ls_keys, ldf_data))

na_price = d_data['close'].values
na_normalized_price = na_price / na_price[0, :]


mod =[]

for i in range(0,len(na_price)):
    mod.append(np.append(na_price[i],sum(p*q for p,q in zip(res.x, na_price[i]))))

first = mod[0].copy()

for i in range(0,len(mod)):
    for j in range(0,n+1):
        mod[i][j] = mod[i][j] / first[j]


plt.clf()

ls_symbols.append('Sharpe')
plt.plot(ldt_timestamps, mod)
plt.legend(ls_symbols)
plt.ylabel('Adjusted Close')
plt.xlabel('Date')
plt.savefig('after.pdf', format='pdf')
