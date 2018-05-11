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
                     'tumor_size',
                     'inv_nodes',
                     'node_caps',
                     'deg_malig',
                     'breast',
                     'breast_quad',
                     'irradiat',
                 ])
# print(df.head())
# print(df.info())

# X = df.drop(['class'], axis=1)
# Y = df['class']


print(df.dtypes)




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
del age_mapping

tumor_size_mapping = [
    '0-4',
    '5-9',
    '10-14',
    '15-19',
    '20-24',
    '25-29',
    '30-34',
    '35-39',
    '40-44',
    '45-49',
    '50-54',
    '55-59'
]

df = label_encoding(df, 'tumor_size', tumor_size_mapping)
del tumor_size_mapping

inv_nodes_mapping = [
    '0-2',
    '3-5',
    '6-8',
    '9-11',
    '12-14',
    '15-17',
    '18-20',
    '21-23',
    '24-26',
    '27-29',
    '30-32',
    '33-35',
    '36-39'
]
df = label_encoding(df, 'inv_nodes', inv_nodes_mapping)
del inv_nodes_mapping

print(df.head())
# binary encode

class_mapping = [
    'no-recurrence-events',
    'recurrence-events'
]

df = label_encoding(df, 'class', class_mapping)
del class_mapping

node_caps_mapping = [
    'yes',
    'no'
]
print(df.head())
df['node_caps'] = df['node_caps'].replace('?','no')
df = label_encoding(df, 'node_caps', node_caps_mapping)
del node_caps_mapping

breast_mapping = [
    'left',
    'right'
]

df = label_encoding(df, 'breast', breast_mapping)
del breast_mapping

irradiat_mapping = [
    'yes',
    'no'
]

df = label_encoding(df, 'irradiat', irradiat_mapping)
del irradiat_mapping


print(df.head())
print(df.dtypes)

# print(df[df['node_caps'] =='?']['node_caps'].describe())
# print(df['node_caps'].describe())

# print(df['node_caps'].describe())

# print(df['breast_quad'].describe())
df['breast_quad'] = df['breast_quad'].replace('?', 'left_low')
# print(df['breast_quad'].describe())
def one_hot_encoding(dataframe, column_name):
    one_hot = pd.get_dummies(dataframe[column_name])
    dataframe = dataframe.drop(column_name, axis=1)
    dataframe = dataframe.join(one_hot)
    return dataframe

df = one_hot_encoding(df, 'breast_quad')
df = one_hot_encoding(df, 'menopause')

print(df.head())
print(df.dtypes)

X = df.drop(['class'], axis=1)
Y = df['class']