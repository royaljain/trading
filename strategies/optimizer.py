# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 19:04:17 2015

@author: royal
"""
import numpy as np
import pandas as pd

from scipy.optimize import minimize, rosen, rosen_der


n =  int(input('Enter number :'))

e = []
f = []


for i in range(0,n):
    sym =  input('Enter symbol :')
    quote = "quotes_" + sym
    df = pd.read_csv('C:\\Users\\royal\\prod\\MarketYahoo\\quotes\\'+quote+'.txt', header=0)
    e.append(df['Close'][0:250].mean())
    f.append(df['Close'].values[0:250])


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
bnds = (0,1)
for i in range(0,n-1):
   bnds = bnds,(0,1)

l = 1
for i in range(0,n-1):
   l = l,0

l = [l]

res = minimize(fun,l, method='SLSQP', bounds=bnds,constraints=cons)