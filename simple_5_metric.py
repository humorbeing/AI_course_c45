import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import numpy as np
from sklearn.metrics import accuracy_score
from me_made_module import me
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

X_train, X_test, Y_train, Y_test = me.get_split(0.224)

rac_record = []
ps_record = []
rs_record = []
f1_record = []
ac_record = []
for model in range(200):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)

    y_test_predict = clf.predict(X_test)
    y_test_proba = clf.predict_proba(X_test)

    y_test = Y_test.tolist()

    y_test_proba =np.array(y_test_proba)
    y_test_proba = y_test_proba.T
    y_test_proba = y_test_proba[1]


    rac = roc_auc_score(y_test, y_test_proba)
    rac_record.append(rac)


    ps = precision_score(y_test, y_test_predict)
    ps_record.append(ps)


    rs = recall_score(y_test, y_test_predict)
    rs_record.append(rs)

    f1 = f1_score(y_test, y_test_predict)
    f1_record.append(f1)

    accuracy = accuracy_score(y_test, y_test_predict)
    ac_record.append(accuracy)

rac_mean = np.mean(rac_record)
ps_mean = np.mean(ps_record)
rs_mean = np.mean(rs_record)
f1_mean = np.mean(f1_record)
ac_mean = np.mean(ac_record)

print('rac:', rac_mean)
print('ps:', ps_mean)
print('rs:', rs_mean)
print('f1:', f1_mean)
print('ac:', ac_mean)
