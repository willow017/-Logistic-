
  # 基于Logistic回归模型的航班数据分析与预测

本次数据分析题目地址：[2022年首届钉钉杯大学生大数据挑战赛初赛B：航班数据分析与预测](https://github.com/willow017/-Logistic-/blob/main/%E5%88%9D%E8%B5%9BB%EF%BC%9A%E8%88%AA%E7%8F%AD%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E4%B8%8E%E9%A2%84%E6%B5%8B.pdf)

### 针对问题一
  本文采用Python编程语言，在给定航班出发机场、目的机场、出行时间等数据信息的前提下，对原始数据进行统计以及数据融合、数据清洗、数据转换，然后进行数据分析，采用Dijkstra算法，将算法中原本的距离替换为数据中的航班飞行时长，从而构建航班转机功能，确定时间最短的航班转机方案。最终根据题目要求，查询到2003年7月4日出发，2003年7月5日到达，从CVG机场到ANC机场的最短时间转机方案为CVG-ORD-ANC。  
  问题一的完整代码：[航班转机功能实现](https://github.com/willow017/-Logistic-/blob/main/ddb1.py)
### 针对问题二
  本文先明确航班延误的含义，分两种情况进行研究。对附件给定的数据进行统计以及数据预处理，展开数据分析，确定对航班延误预测影响的特征，从而进行数据建模。建立起分别以影响航班是否延误以及延误原因的指标作为自变量，以是否延误为因变量的Logistic回归分析模型。然后对Logistic回归分析模型进行训练，再将测试集特征集带入训练完成的预测模型中，得到预测结果。训练过的模型可以预测出近100%的准确率，表现出了较高的预测精度，可以为航班延误的预测问题提供较为准确的参考。  
  问题二的完整代码：[航班延误准确率](https://github.com/willow017/-Logistic-/blob/main/ddb2.py)
  
## 训练集中航班延误比例
 ![](https://github.com/willow017/-Logistic-/blob/main/%E8%AE%AD%E7%BB%83%E9%9B%86%E8%88%AA%E7%8F%AD%E5%BB%B6%E8%AF%AF.png)
 
## 训练集中航班延误原因比例
 ![](https://github.com/willow017/-Logistic-/blob/main/%E8%AE%AD%E7%BB%83%E9%9B%86%E8%88%AA%E7%8F%AD%E5%BB%B6%E8%AF%AF%E5%8E%9F%E5%9B%A0.png)



