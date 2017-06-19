import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import KFold 

df_feature = pd.read_csv('./data/train_data.csv')
df_test = pd.read_csv('./data/test_data.csv')
df_labels = pd.read_csv('./data/train_labels.csv')
df_id = pd.read_csv('./data/test_id.csv')
labels = np.array(df_labels['labels'])

# 随机森林和GBDT模型
clf_rf = RandomForestClassifier(n_estimators=20)
clf_grd = GradientBoostingClassifier(n_estimators=20, random_state = 10)

# 数据标准化
sc = StandardScaler().fit(df_feature)
train = sc.transform(df_feature)
test = sc.transform(df_test)
'''
x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state=42)
print("Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test)))

clf_rf.fit(x_train, y_train)
y_pred_rf = clf_rf.predict(x_test)
print('RF correct prediction: {:4.4f}'.format(np.mean(y_pred_rf == y_test)))
print(metrics.classification_report(y_test, y_pred_rf, target_names=['0','1']))
confusion_matrix_rf = metrics.confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:")
print(confusion_matrix_rf)

clf_grd.fit(x_train, y_train)
y_pred_grd = clf_grd.predict(x_test)
print('GRD correct prediction: {:4.4f}'.format(np.mean(y_pred_grd == y_test)))
print(metrics.classification_report(y_test, y_pred_grd, target_names=['0','1']))
confusion_matrix_grd = metrics.confusion_matrix(y_test, y_pred_grd)
print("Confusion Matrix:")
print(confusion_matrix_grd)
'''
# 官方分数计算
def eval_score(confusion_matrix):
	precision = float(confusion_matrix[0][0]) / (confusion_matrix[0][0] + confusion_matrix[1][0])
	recall = float(confusion_matrix[0][0]) / (confusion_matrix[0][0] + confusion_matrix[0][1])
	f_score = float(5 * precision * recall) / (2 * precision + 3 * recall) * 100
	return f_score

# 使用新的评价函数进行交叉验证
def score_cal(df, label, model, n_folds = 10):
    kf = KFold(df.shape[0], n_folds, shuffle = True)
    scores = []
    for train, test in kf:
        x_train = df.loc[train,:]
        x_test = df.loc[test,:]
        y_train = np.array(label[train]).ravel()
        y_test = np.array(label[test]).ravel()
        model.fit(x_train, y_train)
        y_pred= model.predict(x_test)
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        f_score_xgb = eval_score(confusion_matrix)
        scores.append(f_score_xgb)
    final_score = np.mean(scores)
    return final_score
'''
score_rf = score_cal(df_feature, labels, clf_rf)
score_grd = score_cal(df_feature, labels, clf_grd)
print(score_rf, score_grd)
'''

# 将所有训练样本都用于建模，对10W个样本做预测，并生成能够提交的文档
clf_rf.fit(train, labels)  
predict = clf_rf.predict(test)
result = df_id
result.loc[:,'predict'] = predict
submit = result.loc[result['predict'] == 0]
filepath = './result/'
filename = filepath + 'BDC0564_' + str(datetime.now()).split(' ')[0].replace('-','') + '.txt'
submit['id'].to_csv(filename, header=None, index = False)
