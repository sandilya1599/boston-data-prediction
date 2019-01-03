#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 22:36:40 2019

@author: sandilya
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
dataset=pd.read_csv("train.csv")
x_train,y_train=dataset.iloc[:,1:-1],dataset.iloc[:,-1]
test_set=pd.read_csv("test.csv")
x_test=test_set.iloc[:,1:]
y_test=pd.read_csv("submission.csv")
dec=LinearRegression().fit(x_train,y_train)
y_test['medv']=dec.predict(x_test)
y_test.to_csv("solution.csv",encoding="utf-8",index=False)
#print(len(x_test),len(y_test))