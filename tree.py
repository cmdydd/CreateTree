import pandas as pd
from pylab import mpl
from sklearn import model_selection
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
# mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']# 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


train = pd.read_csv('第五次实验A类分选总数据.csv',engine='python',
usecols=['TOA','PA','PW','RF_START','DOA','category'])
train2 = pd.read_csv('第五次实验B类分选总数据.csv',engine='python',
usecols=['TOA','PA','PW','RF_START','DOA','category'])
train3 = pd.read_csv('第五次实验C类分选总数据.csv',engine='python',
usecols=['TOA','PA','PW','RF_START','DOA','category'])


train = train.append(train2)
train = train.append(train3)


X = train
y = X.pop('category')
 # 把x中的标签category拿出去给y
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.99999,
                                                                    #random_state=42
                                                                    )
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.,
                                                                    #random_state=42
                                                                    )

# print("ceshi",X_train)
# print(X_train)
# print(X_test)
# print(y_train)
print(y_test)


model = DecisionTreeClassifier()
# model=XGBClassifier()


model.fit(X_train,y_train)



accuracy = model.score(X_test, y_test)


# print("acc:", accuracy)
y_pred =model.predict(X_test)
# print('F1 score:', f1_score(y_test, y_pred,average='weighted'))
# print('Recall:', recall_score(y_test, y_pred,
#                               average='weighted'))
# print('Precision:', precision_score(y_test, y_pred,
#                                     average='weighted'))




