import pandas as pd
import numpy as np
from collections import Counter
import pickle

testfile = './data/dsjtzs_txfz_test1.txt'
trainfile = './data/dsjtzs_txfz_training.txt'

def read_data(file):
	df = pd.read_csv(file, sep=' ', header = None)
	if len(df.columns) > 3:
		df = df.rename(columns = {0:'id',1:'movement',2:'target',3:'labels'})
		labels = list(df['labels'])
		del df['id']
		del df['labels']
		return df, labels
	else:
		df = df.rename(columns = {0:'id',1:'movement',2:'target'})
		ids = df['id']
		del df['id']
		return df, ids	

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
	del df['target']
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
		track[index] = d
	return df, track, movements

# 计算两点之间的距离
def dist(p1, p2):
	diff_x = p1[0] - p2[0]
	diff_y = p1[1] - p2[1]
	return np.sqrt(diff_x ** 2 + diff_y ** 2)

# 计算三点间的曲率
def curve_rate(p1,p2,p3):
	d = dist(p1,p3)
	if d != 0:
		return (dist(p1,p2) + dist(p2,p3)) / d
	else:
		return 1.0
    
# 计算两点间与正北方向的夹角
def arct(p1,p2):
	diff = np.abs(np.array(p2) - np.array(p1))
	if diff[0] != 0:
		return np.arctan(diff[1] / diff[0]) * 180 / np.pi
	else:
		return 0.0

def get_features(track):
	features_list = []
	for _,v in track.items():
		features = []
		x = v['x']
		y = v['y']
		t = v['t']
		length  = len(x)
		# 去除瞬移的点
		outlier_index = [i for i in range(length) if t[i-1] == t[i]]
		coord = [[x[i], y[i]] for i in range(length) if i not in outlier_index]
		t = [t[i] for i in range(length) if i not in outlier_index]

		# 距离，时间间隔
		distance, time, velocity = [], [], [0]
		for i in range(1,len(coord)):
			d = dist(coord[i-1], coord[i])
			delta_t = t[i] - t[i-1]
			distance.append(d)
			time.append(delta_t)
		duration = np.sum(time)
			
		# 速度
		for i in range(len(time)):
			velocity.append(distance[i] / time [i])
			
		# 平均速度和平均速率
		total_dist = np.sum(distance)
		avg_speed = total_dist / duration
		avg_v = np.mean(velocity)

		# 加速度和曲率
		acceleration = [(velocity[i] - velocity[i-1]) / time[i-1] for i in range(1,len(velocity))]	
		curve = [curve_rate(coord[i-1], coord[i], coord[i+1]) for i in range(1,len(coord)-1)]

		# 方向和转角
		angles = [arct(coord[i-1], coord[i]) for i in range(1,len(coord))]
		direction = 90 - np.array(angles)
		heading = [np.abs(direction[i] - direction[i-1]) for i in range(1,len(direction))]

		# 统计保持水平或垂直的次数以及时间比例
		keep_direction = Counter(direction)[90]
		keep_index = [i for i,v in enumerate(list(direction)) if int(v) == 90]
		keep_time = np.sum(time[index] for index in keep_index) / duration

		feature = [distance, time, acceleration, curve, direction, heading]
		feature_mean = list(map(lambda x: np.mean(x), feature))
		feature_std = list(map(lambda x: np.std(x), feature))
		features = [duration, keep_direction, keep_time, avg_v, avg_speed, total_dist]
		for f in feature_mean:
			features.append(f)
		for f in feature_std:
			features.append(f)	
		features_list.append(features)

	column_names = ['duration', 'keep_direction', 'keep_time', 'avg_v', 'avg_speed', 'total_distance',\
					'mean_dist','mean_time', 'mean_acc', 'mean_curve', 'mean_dir', 'mean_head', \
					'std_dist','std_time', 'std_acc', 'std_curve', 'std_dir', 'std_head']
	df_feature = pd.DataFrame(features_list)
	assert len(df_feature.columns) == len(column_names)
	df_feature.columns = column_names

	return df_feature

df_train, labels = read_data(trainfile)
df_train, track_train, movements_train = data_reshape(df_train)
output = open('data.pkl', 'wb')
pickle.dump(track_train, output)
print('step 0 ok')
df_test, id_list = read_data(testfile)
df_test, track_test, movements_test = data_reshape(df_test)
print('step 1 ok')

# 去除异常数据
normal_index_train = [i for i in track_train if len(track_train[i]['x']) > 3]
#outlier_index_train = [i for i in range(len(track_train)) if i not in normal_index_train]
normal_index_test = [i for i in track_test if len(track_test[i]['x']) > 3]
#outlier_index_test = [i for i in range(len(track_test)) if i not in normal_index_test]
labels = [labels[i] for i in normal_index_train]
id_list = [id_list[i] for i in normal_index_test]
track_train = {i:track_train[i] for i in normal_index_train}
track_test = {i:track_test[i] for i in normal_index_test}
print('step 2 ok')

df_feature = get_features(track_train)
print('step 3 ok')
df_test = get_features(track_test)
print('step 4 ok')
df_labels = pd.DataFrame(labels)
df_labels.columns = ['labels']
df_id = pd.DataFrame(id_list)
df_id.columns = ['id']

df_feature.to_csv('./data/train_data.csv', index = False)
df_labels.to_csv('./data/train_labels.csv', index = False)
df_test.to_csv('./data/test_data.csv', index = False)
df_id.to_csv('./data/test_id.csv', index = False)