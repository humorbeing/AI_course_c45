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

X_train, X_test, Y_train, Y_test = me.get_split(0.224)


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

average_precision = average_precision_score(y_train, y_train_proba)
print('Average precision on trainset:',average_precision)
average_precision = average_precision_score(y_test, y_test_proba)
print('Average precision on testset:',average_precision)


accuracy = accuracy_score(y_train, y_train_predict)
print('Accuracy on trainset:', accuracy)
accuracy = accuracy_score(y_test, y_test_predict)
print('Accuracy on testset:', accuracy)


precision, recall, _ = precision_recall_curve(y_test, y_test_proba)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
# plt.show()

logl = log_loss(y_train, y_train_proba)
print('Log loss on trainset:', logl)
logl = log_loss(y_test, y_test_proba)
print('Log loss on testset:', logl)


from sklearn.metrics import f1_score

f1 = f1_score(y_train, y_train_predict, average='binary')
print('F1 score on trainset:', f1)
f1 = f1_score(y_test, y_test_predict, average='binary')
print('F1 score on testset:', f1)


from sklearn.metrics import precision_score

ps = precision_score(y_train, y_train_predict)
print('Precision score on trainset:', ps)
ps = precision_score(y_test, y_test_predict)
print('Precision score on testset:', ps)

from sklearn.metrics import recall_score

rs = recall_score(y_train, y_train_predict)
print('Recall score on trainset:', rs)
rs = recall_score(y_test, y_test_predict)
print('Recall score on testset:', rs)

from sklearn.metrics import hamming_loss

hl = hamming_loss(y_train, y_train_predict)
print('Hamming loss on trainset:', hl)
hl = hamming_loss(y_test, y_test_predict)
print('Hamming loss on testset:', hl)


from sklearn.metrics import hinge_loss

hinl = hinge_loss(y_train, y_train_proba)
print('Hinge loss on trainset:', hinl)
hinl = hinge_loss(y_test, y_test_proba)
print('Hinge loss on testset:', hinl)

from sklearn.metrics import roc_auc_score

rac = roc_auc_score(y_train, y_train_proba)
print('Roc auc score on trainset:', rac)
rac = roc_auc_score(y_test, y_test_proba)
print('Roc auc score on testset:', rac)