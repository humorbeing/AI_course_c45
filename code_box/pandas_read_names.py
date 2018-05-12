import sys
sys.path.insert(0, '../')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# sns.countplot(df['age'])
# sns.countplot(df['menopause'])
# sns.countplot(df['tumor_size'])
# sns.countplot(df['breast'])
# sns.countplot(df['deg_malig'])
# sns.countplot(df['node_caps'])
# sns.countplot(df['breast_quad'])
# b = sns.countplot(df['class'])
# b.set_xlabel('X', fontsize=50)
sns.set(font_scale=1.4)
sns.countplot(df['breast_quad'])

plt.show()
