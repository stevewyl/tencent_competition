# tencent_competition
Data and py scripts for tencent compettion

## 运行说明
+ 切换至代码所在文件，在命令行中输入 python model.py
+ xgboost目前只在python2上成功运行，所以得在python2的环境下运行
+ 运行所需的包：bokeh, sklearn

## 数据说明
训练数据表名称：dsjtzs_txfz_training
数据字段：
1. 编号id
2. 鼠标移动轨迹(x,y,t):(x轴位移， y轴位移，在该点的时间戳)
3. 目标坐标(x,y)
4. 类别标签：1-正常轨迹，0-机器轨迹
  
初赛测试表名称：dsjtzs_txfz_test1
复赛测试表名称：dsjtzs_txfz_test2
数据字段：
1. 编号id
2. 鼠标移动轨迹(x,y,t):(x轴位移， y轴位移，在该点的时间戳)
3. 目标坐标(x,y)
  
## 测评标准
最终得分F = 5PR/(2P+3R)*100

## 目前的问题
1. 不太能理解目标坐标的含义，与移动轨迹的坐标差距较大
2. 存在时间t不变的情况，即瞬移的点，暂时将时间间隔从0改为1

## 已加入的特征
1. x_mean: x坐标的平均值
2. y_mean: y坐标的平均值
3. x_std: x坐标的标准差值
4. y_std: y坐标的标准差值
5. diff_mean_x: x坐标移动偏移量的平均值
6. diff_mean_y: y坐标移动偏移量的平均值
7. diff_mean_t: 每次移动所需要时间的平均值
8. v_x_mean: x坐标移动速度的平均值
9. v_y_mean: y坐标移动速度的平均值
10. duration: 从开始移动到结束移动的持续时间
11. slope_mean: 移动轨迹斜率的均值，对于垂直方向的斜率为Inf，方便计算所以默认为0
12. distance：起点位置和终点位置的距离
13. total_distance: 移动轨迹的总长
14. acceleration: 加速度的平均值
15. num_keep_time: x和y坐标保持不变的状态的次数
16. ratio_time_x: x坐标保持不变的状态的时间占比
17. ratio_time_y: y坐标保持不变的状态的时间占比
18. num_movements: 每个样本的移动次数
19. taaget_x: 目标坐标的x值
20. taaget_y: 目标坐标的y值

## 未加入的特征
1. 拐点次数
2. 接近于90度斜率次数
3. 轨迹面积
4. 最大速度和最小速度

## 下一步工作
1. 新的特征提取
2. 模型融合
3. 将数据清理和特征提取与模型代码分开 ---> data_helper.py

## 网友的疑问
1. 目标坐标是指最后一步点击确定的坐标吗？
2. 目标点和轨迹之间的关系，似乎目标点不在轨迹上
3. (x,y,t)轨迹数组，正常情况下t应该是递增的，而数据里t是非递减的（存在t[i] = t[i + 1]的情况），请问是不是数据的问题，还是说这就是正常情况
4. 尽可能在不涉密的情况下，说明一下(x,y,t)是在什么情况下产生的？ 比如说是固定时间抽样？每次鼠标点击？还是其他情况？
5. 有的id轨迹数据点个数只有一个，是因为抽样，还是本身就只有一个或者很少
6. 尽可能在不涉密的情况下，说明是否会在之后给出的10w个测试集数据中，应用新的机器产生鼠标轨迹数据的算法？现在的3k样本数据中是否已经包含了10w测试数据中的所有用到的机器产生数据算法产生的负样本数据？
7. 出现的时间相同，坐标不同，这是采集的问题么？
8. 样本中存在这种现象：Y轴方向位移不变，X轴方向位移在增加，而时间t却在减小。请问这是什么原因产生的？ 比如第814个样本，第53个轨迹点至第54个轨迹点 269    2555    7246 283    2555    277 水平X轴位移增加量283-269，时间t减少7246-277！ 第966个样本也存在这种情况，等等……
