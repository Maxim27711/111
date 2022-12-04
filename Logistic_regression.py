import pandas as pd
import numpy as np
import sklearn.svm
from pandas import read_csv, DataFrame
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


#Первичная Обработка Dataframe:

#считываем файл в DataFrame
df = pd.read_csv("ebw_data.csv")
itog_val = {}
#print(df.describe())
#Расммотрим корреляционную зависимость
#print(df.head())
#print(df.corr(method='pearson'))
#разделяем DataFrame на входные и искомые данные
df_Param = df[['Depth','Width']]
df_data = df.drop(['Depth','Width'], axis=1)
#Делим данные на тренировочные и тестируемые
Xtrn, Xtest, Ytrn, Ytest = train_test_split(df_data, df_Param, test_size=0.4)
mean = Xtrn.mean(axis=0)
# Стандартное отклонение
std = Xtrn.std(axis=0)
kfold=3
Xtrn -= mean
Xtrn /= std
Xtest -= mean
Xtest /= std
Xtrn=Xtrn.astype('int64')
Xtest=Xtest.astype('int64')
model = LogisticRegression(penalty='l2', tol=0.01).fit(Xtrn,Ytrn)
scores = cross_val_score(model, df_data, df_Param, cv = kfold)
itog_val['LogisticRegression'] = scores.mean()

print(itog_val)

