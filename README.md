# tencent_competition
Data and py scripts for tencent compettion

## 运行说明
切换至data文件夹，在命令行中输入 python model.py

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
