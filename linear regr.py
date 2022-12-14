import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Первичная Обработка Dataframe:

#считываем файл в DataFrame
df = pd.read_csv("ebw_data.csv")
#print(df.describe())
#Расммотрим корреляционную зависимость
#print(df.head())
#print(df.corr(method='pearson'))
#разделяем DataFrame на входные и искомые данные
df_Param = df[['Depth','Width']]
df_data = df.drop(['Depth','Width'], axis=1)
#Делим данные на тренировочные и тестируемые
Xtrn, Xtest, Ytrn, Ytest = train_test_split(df_data, df_Param, test_size=0.4)
#Применяем метод линейной регрессии
regr=LinearRegression().fit(Xtrn,Ytrn)
y_pred = regr.predict(Xtest)
#Оценка модели
print("Оценка модели", regr.score(Xtest,Ytest))
#Вводим дополнительные значения на тест:
new_test = pd.DataFrame([[45,140,8.5,75]], columns=['IW','IF','VW','FP'])
New_pred = regr.predict(new_test)
print(New_pred)

