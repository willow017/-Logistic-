import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import classification_report, accuracy_score

# 画图时显示中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_columns', None)  # 显示所有列

# 机场数据
airport_data = pd.read_csv("./数据集/airports.csv", encoding='unicode_escape', low_memory=False)  # 读取文件
# 天气数据
wheather = pd.read_csv("./数据集/rawweatherdata.csv", encoding='unicode_escape', low_memory=False)  # 读取文件
wheather.columns = ['Year', 'Month', 'Day', '最高气温', '平均气温', '最低气温', '最高露点', '平均露点', '最低露点', '最大湿度', '平均湿度',
                    '最小湿度', '最高海平面气压', '平均海平面气压', '最低海平面气压', '最高能见度', '平均能见度', '最低能见度', '最大风速',
                    '平均风速', '瞬时风速', '降水量', '云量', '活动', '风向', '机场', '机场所在城市']  # 列名
wheather['日期'] = pd.to_datetime(wheather[['Year', 'Month', 'Day']])  # 生成日期
wheather = wheather.drop(columns=['Month', 'Day', '最高气温', '平均气温', '最低气温', '最高露点', '平均露点', '最低露点', '最大湿度', '平均湿度',
                    '最小湿度', '最高海平面气压', '平均海平面气压', '最低海平面气压', '最高能见度', '平均能见度', '最大风速', '机场所在城市',
                    '平均风速', '瞬时风速', '降水量', '云量', '活动', '风向',])  # 删除不需要的列

# 函数名：hb()
# 作用：分别将2001-2003和2004-2005的航班数据合并为一个csv文件
def hb():
    list1_3 = []
    list4_5 = []
    for i in range(1, 4):
        df = pd.read_csv("./数据集/200{0}.csv".format(str(i)), encoding='unicode_escape')  # 读取文件
        list1_3.append(df)  # 将数据添加到数组中
    for x in range(4, 6):
        df1 = pd.read_csv("./数据集/200{0}.csv".format(str(x)), encoding='unicode_escape')  # 读取文件
        list4_5.append(df1)  # 将数据添加到数组中
    result = pd.concat(list1_3)  # 合并文件
    result1 = pd.concat(list4_5)  # 合并文件
    result.to_csv('2001_3.csv', index=False, encoding='unicode_escape')  # 保存合并后的文件
    result1.to_csv('2004_5.csv', index=False, encoding='unicode_escape')  # 保存合并后的文件

# 函数名：wheather_process
# 参数说明：
#        两者只能一个为真
#           train_date:为真时，返回训练集天气数据
#           test_date：为真时，返回测试集天气数据
# 作用：返回天气数据
def wheather_process(train_date = None, test_date = None):
    wheather2001_3 = [] # 2001-2003年天气数据
    wheather2004_5 = [] # 2004-2005年天气数据
    if train_date:
        wheather2001 = wheather.loc[wheather['Year'] == 2001]  # 按条件选择数据
        wheather2002 = wheather.loc[wheather['Year'] == 2002]
        wheather2003 = wheather.loc[wheather['Year'] == 2003]
        wheather2001_3.append(wheather2001)  # 将数据添加到数组中
        wheather2001_3.append(wheather2002)
        wheather2001_3.append(wheather2003)
        wheather2001_3 = pd.concat(wheather2001_3)  # 合并文件
        train_wheather = wheather2001_3.loc[wheather2001_3['机场'] == 'MIA']  # 按条件选择数据
        return train_wheather
    if test_date:
        wheather2004 = wheather.loc[wheather['Year'] == 2004]  # 按条件选择数据
        wheather2005 = wheather.loc[wheather['Year'] == 2005]
        wheather2004_5.append(wheather2004)
        wheather2004_5.append(wheather2005)
        wheather2004_5 = pd.concat(wheather2004_5)  # 合并文件
        test_wheather = wheather2004_5.loc[wheather2004_5['机场'] == 'MIA']  # 按条件选择数据
        return test_wheather

# 函数名：data_process
# 参数说明：
#        两者只能一个为真
#           train_date:为真时，返回训练集数据
#           test_date：为真时，返回测试集数据
#        两者只能一个为真
#           arrive：为真时，以实际到达与预计到达之差统计目标数据
#           take_off：为真时，以实际起飞与预计起飞之差统计目标数据
# 作用：返回特征值和目标值
def data_process(train_date = None, test_date = None, arrive= None, take_off=None):
    target = [] #  目标数据
    L_J = []
    if train_date:  # 按条件选择数据
        flight_data = pd.read_csv('./2001_3.csv', encoding='unicode_escape', low_memory=False)
    if test_date:
        flight_data = pd.read_csv('./2004_5.csv', encoding='unicode_escape', low_memory=False)

    flight_data['Day'] = flight_data['DayofMonth']
    flight_data['日期'] = pd.to_datetime(flight_data[['Year', 'Month', 'Day']])
    # 删除年月日和不需要的数据数据，节约内存
    clean_data = flight_data.drop(columns=['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightNum', 'CRSDepTime', 'Diverted', 'LateAircraftDelay', 'UniqueCarrier', 'Day',
                                           'CRSArrTime', 'FlightNum', 'TailNum', 'ActualElapsedTime', 'AirTime', 'Distance', 'TaxiIn', 'TaxiOut', 'CancellationCode',
                                           'SecurityDelay', 'NASDelay', 'WeatherDelay', 'CarrierDelay'])
    #修改列名
    clean_data.rename(columns={'Cancelled': '航班是否取消', 'Origin': '出发机场', 'Dest': '到达机场', 'DepDelay': '实际起飞与预计起飞之差', 'ArrDelay': '实际到达与预计到达之差',
                               'TaxiIn': '起飞滑行', 'TaxiOut': '降落滑行', 'DepTime': '实际起飞', 'CRSElapsedTime': '预计到达与预计起飞之差', 'ArrTime': '实际到达'}, inplace = True)
    clean_data = clean_data.loc[clean_data['出发机场'] == 'MIA']  # 从数据中筛选出出发机场为MIA，航班没有取消，到达机场为JFK和LAX的数据
    clean_data = clean_data.loc[clean_data['航班是否取消'] == 0]
    clean_data_J = clean_data.loc[clean_data['到达机场'] == 'JFK']
    clean_data_L = clean_data.loc[clean_data['到达机场'] == 'LAX']
    L_J.append(clean_data_J)  # 将数据添加到数组中
    L_J.append(clean_data_L)
    result = pd.concat(L_J)  # 合并文件
    result['实际起飞'] = pd.to_numeric(result['实际起飞'], 'coerce')  # 将此列中不是数字的值转换为NAN
    result['实际到达'] = pd.to_numeric(result['实际到达'], 'coerce')
    result['预计到达与预计起飞之差'] = pd.to_numeric(result['预计到达与预计起飞之差'], 'coerce')
    result['实际到达与预计到达之差'] = pd.to_numeric(result['实际到达与预计到达之差'], 'coerce')
    result['实际起飞与预计起飞之差'] = pd.to_numeric(result['实际起飞与预计起飞之差'], 'coerce')
    result = result[~result['实际起飞'].isnull()]  # 过滤掉缺失值的行
    result = result[~result['实际到达'].isnull()]
    result = result[~result['预计到达与预计起飞之差'].isnull()]
    result = result[~result['实际到达与预计到达之差'].isnull()]
    result = result[~result['实际起飞与预计起飞之差'].isnull()]
    # 将航班信息表与天气表用日期连接起来
    if train_date:
        result = pd.merge(result, wheather_process(train_date=1), on='日期')
    if test_date:
        result = pd.merge(result, wheather_process(test_date=1), on='日期')

    result = result.drop(columns=['出发机场', '到达机场', '航班是否取消', '日期', 'Year', '机场'])  # 删除不需要的列
    # 以实际到达与预计到达之差统计目标数据
    if arrive:
        for i in range(0, len(result)):
            if result.iat[i, 3] > 15:
                target.append(1)  # 将数据添加到数组中
            elif result.iat[i, 3] <= 15:
                target.append(0)  # 将数据添加到数组中
    # 以实际起飞与预计起飞之差统计目标数据
    if take_off:
        for i in range(0, len(result)):
            if result.iat[i, 4] > 15:
                target.append(1)  # 将数据添加到数组中
            elif result.iat[i, 4] <= 15:
                target.append(0)  # 将数据添加到数组中
    result['实际起飞'] = result['实际起飞'].astype('int64')  # 数据类型转换
    result['实际到达'] = result['实际到达'].astype('int64')
    result['预计到达与预计起飞之差'] = result['预计到达与预计起飞之差'].astype('int64')
    result['实际到达与预计到达之差'] = result['实际到达与预计到达之差'].astype('int64')
    result['实际起飞与预计起飞之差'] = result['实际起飞与预计起飞之差'].astype('int64')
    result['最低能见度'] = result['最低能见度'].astype('int64')

    train = result.values #将DataFrame类型的数据转换成array类型
    target = np.array(target) #将数组类型转换成array类型
    return train, target

def delayreason_data(train_date = None, test_date = None, arrive= None, take_off=None):
    target = [] #  目标数据
    L_J = []
    if train_date: #  按条件选择数据
        flight_data = pd.read_csv('./2001_3.csv', encoding='unicode_escape', low_memory=False)
    if test_date:
        flight_data = pd.read_csv('./2004_5.csv', encoding='unicode_escape', low_memory=False)
    clean_data = flight_data.drop(columns=['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightNum', 'CRSDepTime', 'Diverted', 'UniqueCarrier', 'ActualElapsedTime', 'DepTime', 'CRSElapsedTime',
                                           'CRSArrTime', 'FlightNum', 'TailNum', 'ActualElapsedTime', 'AirTime', 'Distance', 'TaxiIn', 'TaxiOut', 'CancellationCode', 'ArrTime'])
    #修改列名
    clean_data.rename(columns={'Cancelled': '航班是否取消', 'Origin': '出发机场', 'Dest': '到达机场', 'ArrDelay': '实际到达与预计到达之差',  'DepDelay': '实际起飞与预计起飞之差',
                               'CarrierDelay': '航空公司延误时长', 'WeatherDelay': '天气延误时长', 'NASDelay': '国家航空系统延误时长', 'SecurityDelay': '安全延误时长',
                               'LateAircraftDelay': '晚飞延误时长'}, inplace = True)
    clean_data = clean_data.loc[clean_data['出发机场'] == 'MIA']  # 从数据中筛选出出发机场为MIA，航班没有取消，到达机场为JFK和LAX的数据
    clean_data = clean_data.loc[clean_data['航班是否取消'] == 0]
    clean_data_J = clean_data.loc[clean_data['到达机场'] == 'JFK']
    clean_data_L = clean_data.loc[clean_data['到达机场'] == 'LAX']
    L_J.append(clean_data_J)  # 将数据添加到数组中
    L_J.append(clean_data_L)
    result = pd.concat(L_J)  # 合并文件
    result['航空公司延误时长'] = pd.to_numeric(result['航空公司延误时长'], 'coerce')  # 将此列中不是数字的值转换为NAN
    result['天气延误时长'] = pd.to_numeric(result['天气延误时长'], 'coerce')
    result['国家航空系统延误时长'] = pd.to_numeric(result['国家航空系统延误时长'], 'coerce')
    result['安全延误时长'] = pd.to_numeric(result['安全延误时长'], 'coerce')
    result['晚飞延误时长'] = pd.to_numeric(result['晚飞延误时长'], 'coerce')
    result['实际到达与预计到达之差'] = pd.to_numeric(result['实际到达与预计到达之差'], 'coerce')
    result['实际起飞与预计起飞之差'] = pd.to_numeric(result['实际起飞与预计起飞之差'], 'coerce')
    result = result[~result['航空公司延误时长'].isnull()]  # 过滤掉缺失值的行
    result = result[~result['天气延误时长'].isnull()]
    result = result[~result['国家航空系统延误时长'].isnull()]
    result = result[~result['安全延误时长'].isnull()]
    result = result[~result['晚飞延误时长'].isnull()]
    result = result[~result['实际到达与预计到达之差'].isnull()]
    result = result[~result['实际起飞与预计起飞之差'].isnull()]
    result = result.drop(columns=['出发机场', '到达机场', '航班是否取消'])  # 删除不需要的列
    # 以实际到达与预计到达之差统计目标数据
    if arrive:
        for i in range(0, len(result)):
            if result.iat[i, 0] > 15:
                target.append(1)  # 将数据添加到数组中
            elif result.iat[i, 0] <= 15:
                target.append(0)  # 将数据添加到数组中
    # 以实际起飞与预计起飞之差统计目标数据
    if take_off:
        for i in range(0, len(result)):
            if result.iat[i, 1] > 15:
                target.append(1)  # 将数据添加到数组中
            elif result.iat[i, 1] <= 15:
                target.append(0)  # 将数据添加到数组中
    result['航空公司延误时长'] = result['航空公司延误时长'].astype('int64')  # 数据类型转换
    result['天气延误时长'] = result['天气延误时长'].astype('int64')
    result['国家航空系统延误时长'] = result['国家航空系统延误时长'].astype('int64')
    result['安全延误时长'] = result['安全延误时长'].astype('int64')
    result['晚飞延误时长'] = result['晚飞延误时长'].astype('int64')
    delay_time = result[['航空公司延误时长', '天气延误时长', '国家航空系统延误时长', '安全延误时长', '晚飞延误时长']].sum()  # 计算每种原因的延误总时长
    delay_time = np.array(delay_time)
    pie_reason(delay_time)
    train = result.values #将DataFrame类型的数据转换成array类型
    target = np.array(target) #将数组类型转换成array类型
    return train, target

def pie(array):
    x = []
    for i in np.unique(array):
        x.append(np.sum(array==i))
    x = np.array(x)
    plt.pie(x,
            labels=['未延误飞机', '延误飞机'],  # 设置饼图标签
            colors=['#5d8ca8', "#d5695d"],  # 设置饼图颜色
            autopct='%.2f%%',  # 格式化输出百分比
            )
    plt.title("训练集航班延误")
    plt.savefig('训练集航班延误.png')
    plt.show()

def pie_reason(array):
    plt.pie(
        array,
        labels=['航空公司延误时长', '天气延误时长', '国家航空系统延误时长', '安全延误时长', '晚飞延误时长'],  # 设置饼图标签
        colors=["#d5695d", "#5d8ca8", "#65a479", "#a564c9", '#FF0000'],  # 设置饼图颜色
        autopct='%.2f%%',  # 格式化输出百分比
    )
    plt.title("训练集航班延误原因总时长")
    plt.savefig('训练集航班延误原因.png')
    plt.show()

def delay_model(arrive=None, take_off=None):
    data_train, data_target = data_process(train_date=1, arrive=arrive, take_off=take_off)  # 获取训练集数据
    test_train, test_target = data_process(test_date=1, arrive=arrive, take_off=take_off)  # 获取测试集数据
    pie(data_target)  # 生成延误航班饼形图
    model = LR(max_iter=1000)  # 创建模型,max_iter为训练次数
    model.fit(data_train, data_target)  # 训练模型
    target_pred = model.predict(test_train)  # 预测结果
    target_pred_data = model.predict(data_train)
    print("航班训练集准确率为 %2.3f" % accuracy_score(data_target, target_pred_data))
    print("航班测试集准确率为 %2.3f" % accuracy_score(test_target, target_pred))
    print('航班延误训练集的分数:{0}'.format(model.score(data_train, data_target)))  # 精度
    print('航班延误测试集的分数:{0}'.format(model.score(test_train, test_target)))
    print('航班延误分类报告:')
    print(classification_report(test_target, target_pred))

def delayreason_model(arrive=None, take_off=None):
    data_train, data_target = delayreason_data(train_date=1, arrive=arrive, take_off=take_off)  # 获取训练集数据
    test_train, test_target = delayreason_data(test_date=1, arrive=arrive, take_off=take_off)  # 获取测试集数据
    model = LR(max_iter=1000)  # 创建模型,max_iter为训练次数
    model.fit(data_train, data_target)  # 训练模型
    target_pred = model.predict(test_train)  # 预测结果
    target_pred_data = model.predict(data_train)
    print("航班原因训练集准确率为 %2.3f" % accuracy_score(data_target, target_pred_data))
    print("航班原因测试集准确率为 %2.3f" % accuracy_score(test_target, target_pred))
    print('航班延误原因训练集的分数:{0}'.format(model.score(data_train, data_target)))  # 精度
    print('航班延误原因测试集的分数:{0}'.format(model.score(test_train, test_target)))
    print('航班延误原因分类报告:')
    print(classification_report(test_target, target_pred))

if __name__ == '__main__':
    # 在训练模型前需先运行hb()函数以生成2001_3.csv和2004_5.csv两个文件
    # hb()
    print('以实际到达与预计到达之差统计目标数据:')
    delay_model(arrive=1)
    delayreason_model(arrive=1)
    print('以实际起飞与预计起飞之差统计目标数据')
    delay_model(take_off=1)
    delayreason_model(take_off=1)
