# -*- coding: utf-8 -*-
# (x,y,t):(x轴位移， y轴位移，在该点的时间戳)
# 可能的特征：停留时间的长短，每次位移距离，移动次数，移动轨迹形状（机器的应该比较有规律），相同时间间隔内的轨迹变化
# 问题1 不太能理解目标坐标的含义，与移动轨迹的坐标差距较大
# 问题2 存在时间t不变的情况，即瞬移的点，暂时将时间间隔从0改为1

import pandas as pd
import numpy as np

from bokeh.plotting import figure
from bokeh.io import show, output_file
from bokeh.layouts import gridplot
from bokeh.models import HoverTool

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from datetime import datetime

'''
read data
第一列为ID号，第二列为移动轨迹(x,y,t)，第三列为目标坐标，第四列为标签
'''
test = 'dsjtzs_txfz_test1.txt'
train = 'dsjtzs_txfz_training.txt'

def read_data(file):
	df = pd.read_csv(file, sep=' ', header = None)
	if len(df.columns) > 3:
		df = df.rename(columns = {0:'id',1:'movement',2:'target',3:'labels'})
		labels = df['labels']
		del df['id']
		del df['labels']
		return df, labels
	else:
		df = df.rename(columns = {0:'id',1:'movement',2:'target'})
		ids = df['id']
		del df['id']
		return df, ids	


'''
获得每个样本的坐标点和时间戳以及目标坐标,并计算每个样本的移动次数
返回一个dict, 数字表示ID号，x表示该样本的x坐标值
y表示该样本的y坐标值，t表示该样本每个点的停留时间
形如：{1：{'x':[...],'y':[...],'t':[...]},
	  2:{},
	  ...,
	  3000:{}}
'''
def data_reshape(df):
	num_samples = df.shape[0]
	movements = []
	target_x, target_y = [], []
	for i in range(num_samples):
		movements.append(df['movement'][i].split(';'))
		target_x.append(df['target'][i].split(',')[0])
		target_y.append(df['target'][i].split(',')[1])
	num_movements = [len(m[:-1]) for m in movements]
	df.loc[:,'num_movements'] = num_movements
	df['num_movements'] = df['num_movements'].astype(np.float64)
	df.loc[:,'target_x'] = target_x
	df.loc[:,'target_y'] = target_y
	df['target_x'] = pd.to_numeric(df['target_x'])
	df['target_y'] = pd.to_numeric(df['target_y'])	
	track = {}
	for index,m in enumerate(movements):
		x, y, t = [], [], []
		for point in m[:-1]:
			p = point.split(',')
			one_point = [int(i) for i in p]
			x.append(one_point[0])
			y.append(one_point[1])
			t.append(one_point[2])
		d = {}
		d['x'] = x
		d['y'] = y
		d['t'] = t
		track[index+1] = d
	return df, track, movements

'''
计算保持x,y坐标不变的状态的次数和对应的下标
'''
def cal_keep(diff):
	index = [i for i,v in enumerate(diff) if v == 0]
	keep = []
	v = []
	for i in range(1, len(index)):
		if index[i] - index[i-1] == 1:
			v.append(index[i-1])
			v.append(index[i])
		else:
			v.append(index[i-1])
			v = list(set(v))
			keep.append(v)
			v = []
		if i == len(index) - 2:
			keep.append(v)
	keep_num = len(keep)
	return keep_num, keep

'''
计算保持x,y坐标不变的状态的时间占比
'''
def cal_keep_time(t, keep, total_time):
	begin_end = [[k[0],k[-1]] for k in keep]
	begin_end = [list(set(k)) for k in begin_end]
	time = 0
	if total_time == 0: total_time = 1
	for item in begin_end:
		if len(item) != 1: 
			time += t[item[1]+1] - t[item[0]]
		else:
			time += t[item[0]+1] - t[item[0]]
	ratio = time / total_time
	return ratio

'''
get features
x_mean: x坐标的平均值
y_mean: y坐标的平均值
x_std: x坐标的标准差值
y_std: y坐标的标准差值
diff_mean_x: x坐标移动偏移量的平均值
diff_mean_y: y坐标移动偏移量的平均值
diff_mean_t: 每次移动所需要时间的平均值
v_x_mean: x坐标移动速度的平均值
v_y_mean: y坐标移动速度的平均值
duration: 从开始移动到结束移动的持续时间
slope_mean: 移动轨迹斜率的均值，对于垂直方向的斜率为Inf，方便计算所以默认为0
distance：起点位置和终点位置的距离
total_distance: 移动轨迹的总长
acceleration: 加速度的平均值
num_keep_time: x和y坐标保持不变的状态的次数
ratio_time_x: x坐标保持不变的状态的时间占比
ratio_time_y: y坐标保持不变的状态的时间占比
'''
def get_fetures(df, track):
	# 生成一些空的列表，用于存放新的特征
	x_mean, y_mean = [], []
	x_std, y_std = [], []
	diff_mean_x, diff_mean_y, diff_mean_t = [], [], []
	v_x_mean, v_y_mean = [], []
	duration = []
	slope_mean = []
	d = []
	total_distance = []
	acceleration = []
	keep_nums = []
	ratio_time_x, ratio_time_y = [], []

	for k,v in track.items():
		# 求和或取平均或去标准差作为该样本的特征值
		x_mean.append(np.mean(v['x']))
		y_mean.append(np.mean(v['y']))
		x_std.append(np.std(v['x']))
		y_std.append(np.std(v['y']))
		total_time = v['t'][-1] - v['t'][0]
		duration.append(total_time) 
		d.append(np.sqrt((v['x'][-1] - v['x'][0])**2 + (v['y'][-1] - v['y'][0])**2))

		diff_x, diff_y, diff_t = [], [], []
		slope = []
		# 计算x,y,t的偏移量和斜率
		for i in range(len(v['x'])-1):
			x_diff = v['x'][i+1] - v['x'][i] #计算x坐标的位置偏移量
			y_diff = v['y'][i+1] - v['y'][i] 
			diff_x.append(x_diff)
			diff_y.append(y_diff)
			diff_t.append(np.abs(v['t'][i+1]-v['t'][i])) # 数据异常：有些样本中存在t的差值为负数，故取绝对值
			if x_diff == 0:
				slope.append(0)
			else:
				slope.append(y_diff / x_diff)
		diff_t = [t if t != 0 else 1 for t in diff_t]

		# 计算x和y坐标保持不变的状态时间占比和次数
		keep_num_x, keep_x = cal_keep(diff_y)
		keep_num_y, keep_y = cal_keep(diff_y)
		keep_num = keep_num_x + keep_num_y
		ratio_keep_time_x = cal_keep_time(v['t'], keep_x, total_time)
		ratio_keep_time_y = cal_keep_time(v['t'], keep_y, total_time)

		# 计算x,y方向上的速度
		v_x = np.array(diff_x) / np.array(diff_t)
		v_y = np.array(diff_y) / np.array(diff_t)

		# 计算每个点的距离和总速度
		dis, velocity= [], [0] # 设置第一个点的速度为0
		for i in range(len(diff_x)):
			distance = np.sqrt(diff_x[i]**2 + diff_y[i]**2)
			dis.append(distance)
			velocity.append(distance / diff_t[i])
		diff_v = [velocity[i+1] - velocity[i] for i in range(len(velocity)-1)]
		accc = np.array(diff_v) / np.array(diff_t)

		# 求和或取平均作为该样本的特征值
		ratio_time_x.append(ratio_keep_time_x)
		ratio_time_y.append(ratio_keep_time_y)
		keep_nums.append(keep_num)
		acceleration.append(np.mean(accc))
		total_distance.append(np.sum(dis))
		diff_mean_x.append(np.mean(diff_x))
		diff_mean_y.append(np.mean(diff_y))
		diff_mean_t.append(np.mean(diff_t))
		slope_mean.append(np.mean(slope))
		v_x_mean.append(np.mean(v_x))
		v_y_mean.append(np.mean(v_y))

	# 将不同特征值融合到一个dataframe中
	df.loc[:,'duration'] = duration
	df.loc[:,'x_mean'] = x_mean
	df.loc[:,'y_mean'] = y_mean
	df.loc[:,'x_std'] = x_std
	df.loc[:,'y_std'] = y_std
	df.loc[:,'diff_x'] = diff_mean_x
	df.loc[:,'diff_y'] = diff_mean_y
	df.loc[:,'diff_t'] = diff_mean_t
	df.loc[:,'v_x'] = v_x_mean
	df.loc[:,'v_y'] = v_y_mean
	df.loc[:,'slope_mean'] = slope_mean
	df.loc[:,'b_f_distance'] = d
	df.loc[:,'total_distance'] = total_distance
	df.loc[:,'acceleration'] = acceleration
	df.loc[:,'keep_sums'] = keep_nums
	df.loc[:,'ratio_time_x'] = ratio_keep_time_x
	df.loc[:,'ratio_time_y'] = ratio_keep_time_y
	del df['movement']
	del df['target']

	return df

df_train, labels = read_data(train)
df_train, track_train, movements_train = data_reshape(df_train)
df_train = get_fetures(df_train, track_train)

df_test, id_list = read_data(test)
df_test, track_test, movements_test = data_reshape(df_test)
df_test = get_fetures(df_test, track_test)

df_train = df_train.fillna(df_train.mean())
df_test = df_test.fillna(df_test.mean())

'''
classification model RF
主要流程：1.标准化，2.随机森林模型，树个数为20 3.计算混淆矩阵 4.10折交叉验证
'''
x_train, x_test, y_train, y_test = train_test_split(df_train, labels, test_size=0.2, random_state=42)
print("Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test)))
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
clf = RandomForestClassifier(n_estimators=20)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
y_test = np.array(y_test)
print('RF correct prediction: {:4.4f}'.format(np.mean(y_pred == y_test)))
print('\n')
print(metrics.classification_report(y_test, y_pred, target_names=['0','1']))
print('\n')
print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
f_score = 5 * precision * recall / (2 * precision + 3 * recall) * 100
print(confusion_matrix)
print('\n')
clf_rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=20))
rf_10_fold = cross_val_score(clf_rf, df_train, labels, cv=10)
print('RF 10-fold score: {:4.4f}'.format(np.mean(rf_10_fold)))
# 将3000个样本都用于建模，对10W个样本做预测，并生成能够提交的文档
clf_rf.fit(df_train, labels)  
predict = clf_rf.predict(df_test)
result = pd.DataFrame(id_list)
result.loc[:,'predict'] = predict
submit = result.loc[result['predict'] == 0]
filename = '../result/BDC0564_' + str(datetime.now()).split(' ')[0].replace('-','') + '.txt'
submit['id'].to_csv(filename, header=None, index = False)


'''
# num of label 1: 2600
# num of label 0: 400
print(np.mean(num_movements[0:2600])) # label1：81.6
print(np.mean(num_movements[2600:]))  # label0：95.6
print(np.mean(x_mean[0:2600]))		  # label1：923.9
print(np.mean(x_mean[2600:]))         # label0：829.1
print(np.mean(y_mean[0:2600]))        # label1：2556.7
print(np.mean(y_mean[2600:]))         # label0：2405.9
print(np.mean(t_mean[0:2600]))        # label1：2436.5
print(np.mean(t_mean[2600:]))         # label0：4976.5
'''

'''
plot the movements
'''
def plot_movements(movements):
	pic = []
	label1 = np.arange(0,20,1)
	#label0 = np.arange(2600,2620,1)
	for i in label1:
		movement = movements[i]
		x = [point.split(',')[0] for point in movement[:-1]]
		y = [point.split(',')[1] for point in movement[:-1]]
		p = figure(title = "Sample " + str(i+1) + " movements")
		p.line(x, y, line_width = 2)
		p.circle(x, y, fill_color = 'white', size = 10)
		'''
		p.circle(x, y, size=10,
			fill_color='grey', alpha=0.2, line_color=None,
			hover_fill_color='firebrick', hover_alpha=0.5,
			hover_line_color='white')
		hover = HoverTool(tooltips = None, mode = 'vline')
		p.add_tools(hover)
		'''
		#output_file(str(i) + '.html')
		pic.append(p)
	grid = gridplot(pic, ncols=7)
	show(grid)
#plot_movements(movements)