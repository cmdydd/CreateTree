import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False



train1 = pd.read_csv('ceshi1.csv', engine='python',
                     usecols=['TOA', 'PA', 'PW', 'RF_START', 'DOA', 'category'])
train2 = pd.read_csv('ceshi2.csv', engine='python',
                     usecols=['TOA', 'PA', 'PW', 'RF_START', 'DOA', 'category'])
train3 = pd.read_csv('ceshi3.csv', engine='python',
                     usecols=['TOA', 'PA', 'PW', 'RF_START', 'DOA', 'category'])

i = 0
xl = []
yl = []
ret = {}
retAll=train1[0:]
retAll = retAll.append(train2)
retAll = retAll.append(train3)
retY = retAll.pop('category')
retX = retAll
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)
while i < 30000:
    X1 = train1[0:i + 1000]
    X2 = train2[0:i + 1000]
    X3 = train3[0:i + 1000]
    train = X1
    train = train.append(X2)
    train = train.append(X3)
    X = train
    i = i + 1000
    y = X.pop('category')

    # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

    model = RandomForestClassifier()
    model.fit(X, y)
    # model.fit(X_train, y_train)
    accuracy = model.score(retX,retY)
    ret[len(X)] = accuracy
    xl.append(len(X)) 
    yl.append(accuracy)
    y_pred = model.predict(retX)
print(ret)

xll = xl
yll = yl
plt.figure()
print(xll)
print(yll)
plt.plot(xll,yll)
plt.xlabel(u'sjl')
plt.ylabel(u'准确率')
plt.title(u'准确率随数据量变化曲线')
plt.show()
plt.savefig("line.jpg") 


