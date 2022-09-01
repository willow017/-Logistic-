import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

# 画图时显示中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 机场数据
airport_data = pd.read_csv("./数据集/airports.csv", encoding='unicode_escape', low_memory=False)

# 天气数据
wheather = pd.read_csv("./数据集/rawweatherdata.csv", encoding='unicode_escape', low_memory=False)

# pd.set_option('display.max_columns', None) # 显示所有列
# pd.set_option('display.max_rows', None) # 显示所有行
plt.rcParams['savefig.dpi'] = 400 # 图片像素
plt.rcParams['figure.dpi'] = 400 # 分辨率

# 函数名：read_data
# 参数说明：
#        flight_year:要获取的数据年份
# 作用：获取当前年所有飞行数据
def read_data(flight_year):
    year = flight_year.split('-')[0]
    flight_data = pd.read_csv('./数据集/{0}.csv'.format(year), encoding='unicode_escape', low_memory=False)
    return flight_data

# 函数名：data_process
# 参数说明：
#        flight_data:需要清洗的数据
# 作用：数据清理，删除不需要的数据，将英文列名替换为中文列名，方便阅读理解
def data_process(flight_data):
    flight_data['Day'] = flight_data['DayofMonth']
    flight_data['起飞日期'] = pd.to_datetime(flight_data[['Year','Month','Day']])
    # 删除年月日和不需要的数据数据，节约内存
    clean_data = flight_data.drop(columns=['Year', 'Month', 'DayofMonth', 'LateAircraftDelay', 'SecurityDelay', 'NASDelay', 'WeatherDelay', 'CarrierDelay',
                                            'Distance', 'CRSElapsedTime', 'DayOfWeek', 'FlightNum'])
    #修改列名
    clean_data.rename(columns={'Cancelled': '航班是否取消', 'TaxiIn': '飞机起飞时滑行时间', 'TaxiOut': '飞机降落时滑行时间', 'Origin': '出发机场', 'Dest': '到达机场', 'ActualElapsedTime': '实际到达时间与实际起飞时间之差', 'DepTime': '实际起飞时间',
                            'airport': '机场名称', 'UniqueCarrier': 'iata代码', 'AirTime': '空中飞行时间'}, inplace = True)
    return clean_data


# 函数名：departure
# 参数说明：
#        city：城市名
#        state：州名
# 作用：输入地名返回机场缩写
def departure(city, state):
    num = []
    real_num = []
    airports = airport_data.loc[airport_data['state'] == state]
    airports = airports.loc[airport_data['city'] == city]
    # 获取当前城市所有机场缩写
    for i in range(0, len(airports)):
        if airports.iat[i, 0] not in twoday_data['出发机场'].values:
            if airports.iat[i, 0] not in twoday_data['到达机场'].values:
                print('{0}机场不存在'.format(airports.iat[i, 0]))
                h = airports[airports.iata == airports.iat[i, 0]].index.tolist()[0]
                num.append(h)
        else:
            print('{0}机场存在'.format(airports.iat[i, 0]))
            h = airports[airports.iata == airports.iat[i, 0]].index.tolist()[0]
            real_num.append(h)
    airports = airports.drop(num)
    airports_name = airports.iat[0, 0]
    return airports_name

# 函数名：flightday_data
# 参数说明：
#        clean_data：清洗完成的数据
#        outset_time：出发时间
#        arrive_time：到达时间
# 作用：返回时间段内的所有飞行数据
def flightday_data(clean_data, outset_time, arrive_time):
    twoday_data = clean_data.loc[(clean_data['起飞日期'] == '{0}'.format(outset_time)) | (clean_data['起飞日期'] == '{0}'.format(arrive_time))][clean_data['航班是否取消'] == 0]
    return twoday_data

# 函数名：shortest_path
# 参数说明：
#        twoday_data: 出发时间到到达时间段内所有数据
#        outset_name：出发机场
#        arrive_name：到达机场
# 作用：求出当前时间段内最短时间的路径并画图
def shortest_path(twoday_data, outset_name, arrive_name):
    # 调用两天的数据来生成带权有向图
    FG = nx.from_pandas_edgelist(twoday_data, source='出发机场', target='到达机场', edge_attr='实际到达时间与实际起飞时间之差')
    # 画出图
    nx.draw_networkx(FG, with_labels=True, font_size=5, node_size=50, arrows=True)
    # 求出最短路径
    dijpath = nx.dijkstra_path(FG, source='{0}'.format(outset_name), target='{0}'.format(arrive_name), weight='实际到达时间与实际起飞时间之差')
    # 输出最短路径方案
    print('从{0}到{1}的最短飞行时间转机方案为{2}'.format(outset_name, arrive_name, dijpath))
    # 为图片生成标题
    plt.title('{0}到{1}飞行路线有向图'.format(outset_time, arrive_time))
    # 保持图片
    plt.savefig('./{0}-{1}.png'.format(outset_name, arrive_name))
    # 显示图片
    plt.show()


if __name__ == '__main__':
    # name = departure('Wrangell', 'AK')

    outset_time = input('请输入出发时间(xxxx-x-x)：')
    arrive_time = input('请输入到达时间(xxxx-x-x)：')
    outset_state = input('请输入出发州：')
    outset_city = input('请输入出发城市：')
    arrive_state = input('请输入到达州：')
    arrive_city = input('请输入到达城市：')
    flight_data = read_data(outset_time)
    clean_data = data_process(flight_data)
    twoday_data = flightday_data(clean_data, outset_time, arrive_time)
    outset_name = departure(outset_city, outset_state)
    arrive_name = departure(arrive_city, arrive_state)
    shortest_path(twoday_data, outset_name, arrive_name)







