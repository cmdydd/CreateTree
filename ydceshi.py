import pandas as pd
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

i = 0.9
ret = {}
retAll = train1[0:]
retAll = retAll.append(train2)
print(retAll)
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

# print(retX)
print(ret)
