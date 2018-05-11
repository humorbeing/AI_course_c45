import pandas as pd
import numpy as np

dates = pd.date_range('1/1/2000', periods=8)
df = pd.DataFrame(
    np.random.randn(8, 4),
    index=dates,
    columns=['A', 'B', 'C', 'D']
)
'''
df = pd.read_csv(file_location)
df = pd.read_csv(file_location, header=None)
df = pd.read_csv(file_location,
                 names=[
                     'class',
                     'age',
                     'menopause',
                     'tumor-size',
                     'inv-nodes',
                     'node-caps',
                     'deg-malig',
                     'breast',
                     'breast-quad',
                     'irradiat',
                 ])
'''

print('print(df)')
print(df)
print()
print('print(df.head())')
print(df.head())
print()
print('print(df.head(6))')
print(df.head(6))
print()
print('print(df.tail())')
print(df.tail())
X = df.drop(['A'], axis=1)
Y = df['A']
print()
print('X,Y')
print(X.head())
print(Y.head())
joined = X.join(Y)
print()
print('joined X+Y')
print(joined.head())

df = pd.DataFrame(
    {
        'A': ['a','b','c'],
        'B': ['b','a','c']
    }
)
print(df)
one_hot = pd.get_dummies(df['B'])
df = df.drop('B', axis=1)
df = df.join(one_hot)
print(df)


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(
    [
        'a',
        'b',
        'c',
    ]
)
# print(list(le.classes_))
temp = le.transform(df['A'])
temp = pd.DataFrame(temp, columns=['A'])
# print(temp.head())
df = df.drop('A', axis=1)
df = df.join(temp)
print(df)

