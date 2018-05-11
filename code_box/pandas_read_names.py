import sys
sys.path.insert(0, '../')
import pandas as pd
import matplotlib.pyplot as plt

file_location = "/media/ray/D43E51303E510CBC/MyStuff/Workspace/Python/AI_course_c45/dataset/breast_cancer_dataset.txt"

# df = pd.read_csv(file_location)
# df = pd.read_csv(file_location, header=None)
df = pd.read_csv(file_location,
                 names=[
                     'Class',
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
print(df.head())
print(df.info())