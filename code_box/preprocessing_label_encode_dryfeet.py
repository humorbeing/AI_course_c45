import sys
sys.path.insert(0, '../')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import preprocessing

file_location = "/media/ray/D43E51303E510CBC/MyStuff/Workspace/Python/AI_course_c45/dataset/breast_cancer_dataset.txt"

# df = pd.read_csv(file_location)
# df = pd.read_csv(file_location, header=None)
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
# print(df.head())
# print(df.info())

X = df.drop(['class'], axis=1)
Y = df['class']

print(df.head())
# print(df.info())
print(X.head())
# print(X.info())
print(Y.head())
# print(Y.info())
print(df.dtypes)
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X, Y)
le = preprocessing.LabelEncoder()
le.fit(
    [
        '10-19',
        '20-29',
        '30-39',
        '40-49',
        '50-59',
        '60-69',
        '70-79',
        '80-89',
        '90-99'
    ]
)
print(list(le.classes_))
temp = le.transform(df['age'])
temp = pd.DataFrame(temp, columns=['me'])
print(temp.head())
df = df.join(temp)
print(df[['age','me']])