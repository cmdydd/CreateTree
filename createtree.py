from typing import List

import pandas as pd#pandas被使用者import为pd，pandas是panel data的缩写
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.metrics import recall_score
from sklearn.metrics import fscore
from sklearn.metrics import  precision_score
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
from pylab import mpl
from sklearn.tree import DecisionTreeClassifier
#import相当于是载入某个模块的意思，由于模块有很多，如果默认都装在进来的话容易造成名字冲突，和资源消耗，所以要用的时候import
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
# mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']# 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


train = pd.read_csv('ceshi1.csv',engine='python',
usecols=['TOA','PA','PW','RF_START','DOA','category'])#导入测试1的数据6列
train2 = pd.read_csv('ceshi2.csv',engine='python',
usecols=['TOA','PA','PW','RF_START','DOA','category'])#导入测试2的数据6列
train3 = pd.read_csv('ceshi3.csv',engine='python',
usecols=['TOA','PA','PW','RF_START','DOA','category'])#导入测试3的数据6列
#usecols 选取数据的列，如果iris.txt中的前4列，则usecols=(0,1,2,3)。如果取第5列这一列，则usecols=(4,)。这种取单一列容易出问题。

train = train.append(train2)#增加数据
train = train.append(train3)


i = 0
ret = {}
while i < 90000:
    X = train[0 : i+5000]
    i = i + 5000
    y = X.pop('category')
     # 把x中的标签category拿出去给y
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1 )#random_state=42


    model = RandomForestClassifier()
    #model = DecisionTreeClassifier()
    #model=XGBClassifier()
    model.fit(X_train,y_train)

    accuracy = model.score(X_test, y_test)
    #print("当前数据量：",len(X))
    #print("acc:", accuracy)#acc为准确率
    ret[len(X)] = accuracy
    y_pred = model.predict(X_test)#F1 score为精确度和召回率的调和平均数
    #print('F1 score:', f1_score(y_test, y_pred,average='weighted'))
    #print('Recall:', recall_score(y_test, y_pred, average='weighted'))#recall为召回率
    #print('Precision:', precision_score(y_test, y_pred, average='weighted'))#precision为精确度

print(ret)



