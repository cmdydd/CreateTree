import pandas as pd
from pylab import mpl
import numpy as np
import matplotlib.pyplot as plt
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

i = 0.9
xl = []
yl = []
ret = {}
retAll = train1[0:]
retAll = retAll.append(train2)
retAll = retAll.append(train3)
retY = retAll.pop('category')
retX = retAll

while i > 0:

    X_train, X_test, y_train, y_test = model_selection.train_test_split(retX, retY, test_size=i)
    i = i - 0.1
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    accuracy = model.score(retX,retY)
    ret[len(X_train)] = accuracy
    y_pred = model.predict(retX)
    xl.append(len(X_train)) 
    yl.append(accuracy)


# print(retX)
print(ret)

xll = xl
yll = yl
plt.figure()
print(xll)
print(yll)
plt.plot(xll,yll)
plt.xlabel(u'数据量')
plt.ylabel(u'准确率')
plt.title(u'准确率随数据量变化曲线')
plt.show()
plt.savefig("line.jpg") 


