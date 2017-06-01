import pandas as pd 
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import KFold 

df_train = pd.read_csv('./data/train_data.csv')
df_train = df_train.iloc[:,1:21]
labels = pd.read_csv('./data/train_labels.csv')
labels = labels.iloc[:,1:2]

x_train, x_test, y_train, y_test = train_test_split(df_train, labels, test_size=0.2, random_state=42)
print "Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test))

y_test = np.array(y_test).ravel()
y_train = np.array(y_train).ravel()
'''
xg_train = xgb.DMatrix(x_train, label = y_train)
xg_test = xgb.DMatrix(x_test, label = y_test)

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 5
bst = xgb.train(xg_train, num_round, watchlist)
pred = bst.predict(xg_test)

error_rate = np.sum(pred != y_test) / y_test.shape[0]
print error_rate
'''
clf_xgb = make_pipeline(StandardScaler(), xgb.XGBClassifier(n_estimators=25, learning_rate = 0.5))
clf_xgb.fit(x_train, y_train)
y_pred= clf_xgb.predict(x_test)
percision = 1 - float(np.sum(y_pred != y_test)) / y_test.shape[0]
print percision

def eval_score(confusion_matrix):
	precision = float(confusion_matrix[0][0]) / (confusion_matrix[0][0] + confusion_matrix[1][0])
	recall = float(confusion_matrix[0][0]) / (confusion_matrix[0][0] + confusion_matrix[0][1])
	f_score = float(5 * precision * recall) / (2 * precision + 3 * recall) * 100
	return f_score

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
f_score_xgb = eval_score(confusion_matrix)
print f_score_xgb

labels = np.array(labels).ravel()
xgb_10_fold = cross_val_score(clf_xgb, df_train, labels, cv=10, scoring='f1')
print 'XGB 10-fold score: {:4.4f}'.format(np.mean(xgb_10_fold))

def score_cal(df, labels, model, n_folds = 10):
    kf = KFold(df.shape[0], n_folds, shuffle = True)
    scores = []
    for train, test in kf:
        x_train = df_train.loc[train,:]
        x_test = df_train.loc[test,:]
        y_train = np.array(labels[train]).ravel()
        y_test = np.array(labels[test]).ravel()
        model.fit(x_train, y_train)
        y_pred= model.predict(x_test)
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        f_score_xgb = eval_score(confusion_matrix)
        scores.append(f_score_xgb)
    final_score = np.mean(scores)
    return final_score
score_xgb = score_cal(df_train, labels, clf_xgb)