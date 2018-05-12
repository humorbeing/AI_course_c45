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
from sklearn.metrics import make_scorer


X_train, X_test, Y_train, Y_test = me.get_split(0.224)

rac_record = []
ps_record = []
rs_record = []
f1_record = []
ac_record = []
for epoch in range(5):
    tree_para = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': range(1, 30),
        'min_samples_split': [2, 3, 4, 0.3, 0.5, 0.7],
        'min_samples_leaf': [1, 2, 3, 0.3, 0.4, 0.5]
    }

    clf = GridSearchCV(
        tree.DecisionTreeClassifier(),
        tree_para,
        cv=4,
        # scoring=make_scorer(roc_auc_score)
    )
    # clf = tree.DecisionTreeClassifier()
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

    # print('rac:', rac)

print('mean rac:', np.mean(rac_record), 'rac min:', np.min(rac_record))
print('rac max:', np.max(rac_record))
print('ps max:', np.max(ps_record))
print('rs max:', np.max(rs_record))
print('f1 max:', np.max(f1_record))
print('ac max:', np.max(ac_record))

'''
mean rac: 0.5989702517162471 rac min: 0.5617848970251715
rac max: 0.625858123569794
ps max: 0.4375
rs max: 0.3684210526315789
f1 max: 0.39999999999999997
ac max: 0.676923076923077'''