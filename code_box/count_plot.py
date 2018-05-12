import sys
sys.path.insert(0, '../')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
file_location = "/media/ray/D43E51303E510CBC/MyStuff/Workspace/Python/AI_course_c45/dataset/breast_cancer_dataset.txt"
from sklearn import preprocessing

def label_encoding(dataframe, column_name, mapping_list):
    le = preprocessing.LabelEncoder()
    le.fit(mapping_list)
    # print(list(le.classes_))
    temp = le.transform(dataframe[column_name])
    temp = pd.DataFrame(temp, columns=[column_name])
    # print(temp.head())
    dataframe = dataframe.drop(column_name, axis=1)
    dataframe = dataframe.join(temp)
    return dataframe
# df = pd.read_csv(file_location)
# df = pd.read_csv(file_location, header=None)
df = pd.read_csv(file_location,
                 names=[
                     'class',
                     'age',
                     'menopause',
                     'tumor_size',
                     'inv_nodes',
                     'node_caps',
                     'deg_malig',
                     'breast',
                     'breast_quad',
                     'irradiat',
                 ])
df['breast_quad'] = df['breast_quad'].replace('?', 'left_low')
df['node_caps'] = df['node_caps'].replace('?','no')
age_mapping = [
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
df = label_encoding(df, 'age', age_mapping)

def one_hot_encoding(dataframe, column_name):
    one_hot = pd.get_dummies(dataframe[column_name])
    dataframe = dataframe.drop(column_name, axis=1)
    dataframe = dataframe.join(one_hot)
    return dataframe
print(df.head())
df = one_hot_encoding(df, 'breast_quad')
print(df.head())
# sns.countplot(df['age'])
# sns.countplot(df['menopause'])
# sns.countplot(df['tumor_size'])
# sns.countplot(df['breast'])
# sns.countplot(df['deg_malig'])
# sns.countplot(df['node_caps'])
# sns.countplot(df['breast_quad'])
# b = sns.countplot(df['class'])
# b.set_xlabel('X', fontsize=50)
# sns.set(font_scale=1.4)
# sns.countplot(df['age'])
#
# plt.show()
