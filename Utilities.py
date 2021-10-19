# 刘金波
# 开发时间： 2021/10/11 20:01
#coding: UTF-8
# #向csv文件中写入数据
# import csv
# import codecs
# data=[('测试4','测试工程师1'),
#       ('测试2','测试工程师2'),
#       ('测试3','测试工程师3')
#       ]
# f=codecs.open('111.csv','w','gbk')
# write=csv.writer(f)
# for i in data:
#     write=write.writerow(i)
# f.close()
#head_row=next(reader):读取第一行数据
#column=[row[0] for row in reader]:读取第一列数据
#for index,column_header in enumerate(head_row):获取文件头及其索引
#输出函数图表（线段）
# import numpy as np
# from matplotlib import pyplot as plt
# x=np.arange(1,11)
# y=2*x+5
# plt.title('matplotlib demo')
# plt.xlabel('x axis caption')
# plt.ylabel('y axis caption')
# plt.plot(x,y,'ob')#用圆点来代替线段输出图表
# plt.show()
# #输出正弦波
# import numpy as np
# from matplotlib import pyplot as plt
# x=np.arange(0,4*np.pi,0.1)
# y_cos=np.cos(x)
# y_sin=np.sin(x)
# #建立一个subpolt网格，高为2，宽为1，绘制第一个网格
# plt.subplot(2,1,1)
# plt.plot(x,y_sin)
# plt.subplot(2,1,2)
# plt.plot(x,y_cos)
# plt.show()
#生成条形图
# from matplotlib import pyplot as plt
# x=[5,8,10]
# y=[12,16,19]
# x1=[7,9,11]
# y1=[6,8,9]
# plt.bar(x,y,color='g',align='center')#水平垂直居中
# plt.bar(x1,y1,color='r',align='center')
# plt.show()
#histogram
# import numpy as np
# a=np.array([11,33,45,67,88,35,79,4,78,77])
# np.histogram(a,bins=[0,20,40,60,80,100])
# hist,bins=np.histogram(a,bins=[0,20,40,60,80,100])
# print(hist)
# print(bins)
# #plt():建立直方图
# from matplotlib import pyplot as plt
# import numpy as np
# a=np.array([11,33,45,67,88,35,79,4,78,77])
# plt.hist(a, bins =[0,20,40,60,80,100])
# plt.show()
# import torch
# z=torch.ones(2,3,4)
# print(z)
# print(z.size())
# print(z.size(0))
# print(z.size(1))
# print(z,size(2))
#得到最大值的序号索引，行不变，进行列之间的比较
# import torch
# a=torch.rand(3,4)
# print(a)
# b=torch.argmax(a,dim=1)
# print(b)
#求和
# import torch
# a=torch.arange(0,6).view(2,3)
# b=a.sum()
# c=a.sum(dim=0)
# d=a.sum(dim=1)
# print(b)
# print(c)
# print(d)
#累加
# import torch
# a=torch.arange(0,6).view(2,3)
# print(a)
# b=a.cumsum(dim=0)
# c=a.cumsum(dim=1)
# print(b)
# print(c)
import tensorflow as tf



















