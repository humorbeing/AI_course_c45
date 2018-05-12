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

# schedule = [
#     0.15,
#     0.2,
#     0.25,
#     0.3,
#     0.35,
#     0.4,
#     0.45,
#     0.5,
#     0.55,
#     0.6,
#     0.65,
#     0.7
# ]
schedule = np.linspace(0.215, 0.23, 100)
rac_mean = []
ps_mean = []
rs_mean = []
f1_mean = []
ac_mean = []
for s in schedule:
    X_train, X_test, Y_train, Y_test = me.get_split(s)

    # train_record = []
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

        y_train_predict = clf.predict(X_train)
        y_train_proba = clf.predict_proba(X_train)

        y_test = Y_test.tolist()
        y_train = Y_train.tolist()


        y_test_proba =np.array(y_test_proba)
        y_test_proba = y_test_proba.T
        y_test_proba = y_test_proba[1]

        y_train_proba =np.array(y_train_proba)
        y_train_proba = y_train_proba.T
        y_train_proba = y_train_proba[1]


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
    # import seaborn as sns
    # sns.set(color_codes=True)
    # sns.distplot(test_record)
    # sns.distplot(train_record)
    rac_mean.append(np.mean(rac_record))
    ps_mean.append(np.mean(ps_record))
    rs_mean.append(np.mean(rs_record))
    f1_mean.append(np.mean(f1_record))
    ac_mean.append(np.mean(ac_record))
    # plt.show()

plt.plot(schedule, rac_mean, 'y', label='Roc auc score')
plt.plot(schedule, ps_mean, 'r', label='Precision score')
plt.plot(schedule, rs_mean, 'b', label='Recall score')
plt.plot(schedule, f1_mean, 'g', label='F1 score')
plt.plot(schedule, ac_mean, 'k', label='Accuracy score')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('Testset proportion in total dataset')
plt.ylabel('Score')
plt.show()