# coding=utf-8
from pandas import read_csv
from pandas import datetime
from pandas import concat
from pandas import DataFrame
from pandas import Series
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Activation
from keras.layers import Dropout
from math import sqrt
from tensorflow.keras import optimizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from matplotlib import pyplot
import numpy
import tensorflow
import math
import statsmodels.api as sm
import seaborn as sns
from scipy import stats
import pandas
import warnings
import matplotlib.patches as pch
import gc
warnings.filterwarnings("ignore", category=numpy.VisibleDeprecationWarning)

# 读取时间数据的格式化
# def parser(x):
#   return x

train_size = 585350  # 用于[training + validation]的size
train_val_size = 480000 # 用于[training]的size; 可得用于[validation]的size = train_size - train_val_size
# test_size = 10000
batch_size = 512
epoch_size = 1000
neurons_num = 20
difference_order = 1 #差分的阶数，一阶差分就是时间序列的相邻差 x(t)-x(t-1)

n_row = 100 # input的行数
n_col = 100 # input的列数
n_row_col = n_row * n_col # 用过去100*100的时间序列作为input，先按照行排列，再按照列排列
n_len = 40 # 预测未来40个位置, 全部为0，分类为0；只要有一个为1，分类为1
output_size = 2 # 输出[0,1]代表未来40个全部为0，输出[1,0]代表未来40个至少有一个为1

def create_model(n_col, n_row, output_size, n_neurons, batch_size):
    model = Sequential()
    model.add(Bidirectional(LSTM(n_neurons, dropout=0.2, recurrent_dropout=0.2, input_shape=(n_row, n_col), stateful=False, return_sequences=True)))
    # model.add(Bidirectional(LSTM(n_neurons, input_shape=(n_row, n_col), stateful=False, return_sequences=True)))
    # dropout 针对 input x_t 和 hidden layer之间的权重 W_xi, W_xf, W_xo； recurrent_dropout 针对 hidden layer输出h和 hidden layer之间的权重 W_hi, 权重 W_hf, 权重 W_ho
    model.add(Bidirectional(LSTM(n_neurons, input_shape=(n_row, n_col), stateful=False, return_sequences=False)))
    # model.add(Bidirectional(LSTM(n_neurons, batch_input_shape=(batch_size, n_row, n_col), stateful=True, return_sequences=False)))
    # model.add(Dropout(0.1)) # dropout针对最后一个全连接层的权重W_hy
    model.add(Dense(output_size, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # categorical_crossentropy
    # model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model


# 匹配LSTM网络训练数据
def fit_lstm(train_input, train_output, n_col, n_row, output_size, n_neurons, n_batch, nb_epoch):
    # 训练数据格式 [samples, timesteps, features]
    X = train_input
    y = train_output
    y = y.reshape(y.shape[0], y.shape[1])
    # BiLSTM 需要如下转换 X，y原本是int格式
    X = tensorflow.cast(X, tensorflow.bfloat16)
    y = tensorflow.cast(y, tensorflow.bfloat16)
    X = numpy.array(X)
    y = numpy.array(y)

    index_permt, X_epoch, y_epoch, X_train, y_train, X_val, y_val = list(), list(), list(), list(), list(), list(), list()
    index = list()
    index = [i for i in range(len(X))]
    index_permt = numpy.random.permutation(index)
    X_epoch = X[index_permt]
    y_epoch = y[index_permt]
    X_train = X_epoch[0:train_val_size]
    y_train = y_epoch[0:train_val_size]
    X_val = X_epoch[train_val_size:]
    y_val = y_epoch[train_val_size:]
    del index_permt, X_epoch, y_epoch, X, y, index
    gc.collect()
    # # 不做shuffle
    # X_train = X[0:train_val_size]
    # y_train = y[0:train_val_size]
    # X_val = X[train_val_size:]
    # y_val = y[train_val_size:]

    # 配置一个LSTM，参数设置
    model = create_model(n_col, n_row, output_size, n_neurons, n_batch)
    history_loss_list = list()
    history_valloss_list = list()
    history_accuracy_list = list()
    history_valaccuracy_list = list()
    for i in range(0, 1000):
        history = model.fit(X_train, y_train, epochs=1, batch_size=n_batch, verbose=1, shuffle=True, validation_data=(X_val, y_val))
        history_loss_list.append(history.history['loss'])
        history_valloss_list.append(history.history['val_loss'])
        history_accuracy_list.append(history.history['accuracy'])
        history_valaccuracy_list.append(history.history['val_accuracy'])
        # print(history.history['loss'], history.history['val_loss'])
        # print("当前计算次数loss：" + str(i), history.history['loss'], history.history['val_loss'])
        # print("当前计算次数accuracy："+str(i), history.history['accuracy'], history.history['val_accuracy'])
        # # 画出loss，val_loss
        # fig, ax = pyplot.subplots(1, 1)
        # pyplot.plot(history_loss_list)
        # pyplot.plot(history_valloss_list)
        # fig, ax = pyplot.subplots(1, 1)
        # pyplot.plot(history_accuracy_list)
        # pyplot.plot(history_valaccuracy_list)
        model.reset_states()
    model.save_weights('D:/AI_model/Save model/binary_model_16')
    return model

# 预测
def forcast_lstm(model, X):
    yhat = model.predict(X)
    return yhat

# 差分(一阶）
def difference(dataset, interval):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# 逆差分
def inverse_difference(history, yhat, interval, n_seq):  # 历史数据，预测数据，差分间隔
    y_hat = list()
    ############ 1阶-差分预测
    for i in range(0, len(yhat)):
        value = yhat[i] + history[-interval - (n_seq - i - 1)]
        y_hat.append(value)
    yhat = numpy.array(y_hat)
    return yhat

# 加载数据
series = read_csv('D:/AI_model/data_set/UDP_original_onlytime_phy.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)

# 处理多个表头相同的csv级联，合并后save成csv
# series_01 = read_csv('D:/AI_model/data_set/data_set_01/dtfx_file_01.csv', header=0, parse_dates=[0], index_col=0, squeeze=True) # TCP_ceil_del
# series_02 = read_csv('D:/AI_model/data_set/data_set_01/dtfx_file_02.csv', header=0, parse_dates=[0], index_col=0, squeeze=True) # TCP_ceil_del
# series_03 = read_csv('D:/AI_model/data_set/data_set_01/dtfx_file_03.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_04 = read_csv('D:/AI_model/data_set/data_set_01/dtfx_file_04.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_05 = read_csv('D:/AI_model/data_set/data_set_01/dtfx_file_05.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_06 = read_csv('D:/AI_model/data_set/data_set_01/dtfx_file_06.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_07 = read_csv('D:/AI_model/data_set/data_set_01/dtfx_file_07.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# frames = [series_01, series_02, series_03, series_04, series_05, series_06, series_07]
# result = pandas.concat(frames)
# result.to_csv("D:/AI_model/data_set/data_set_01/dtfx_file_all.csv")

# 1ms刻度, 0ms为起始点; 到达时间差平均20ms左右;
raw_values = series.values
raw_values = raw_values * 1000
raw_values = raw_values[2500:26000] # 截取中间一段数据
raw_values = raw_values - raw_values[0] * numpy.ones(len(raw_values))
raw_values = numpy.array(raw_values, int)

diff_values = difference(raw_values, difference_order)  # 转换成 1-order 差分数据
diff_values = diff_values.values
diff_values = numpy.array(diff_values, int)
# fig, ax = pyplot.subplots(1, 1)
# pyplot.plot(diff_values) # 绘制到达时间差，1ms为单位
# print(numpy.mean(diff_values)) # 平均到达时间差，大概~25ms
# print(numpy.sum(diff_values > 40)/len(diff_values)) # 到达时间差，大于40ms占比约~20%
# # 绘制到达时间差的CDF
# ecdf = sm.distributions.ECDF(diff_values)
# fig, ax = pyplot.subplots(1, 1)
# x_x = numpy.linspace(min(diff_values), max(diff_values), 100000)
# y_y = ecdf(x_x)
# pyplot.step(x_x, y_y)

# # 绘制 到达时间差的的PDF histogram
# pyplot.subplot(221)
# pyplot.hist(diff_values)
# # 获得 histogram 数据
# pyplot.subplot(222)
# hist, bin_edges = numpy.histogram(diff_values)
# pyplot.plot(hist)
# # 拟合 fit histogram curve
# pyplot.subplot(223)
# sns.distplot(diff_values, kde=False, fit=stats.gamma, rug=True)
# # 同时绘制 归一化的 PDF + CDF
# pyplot.subplot(224)
# hist, bin_edges = numpy.histogram(diff_values)
# width = (bin_edges[1] - bin_edges[0]) * 0.8
# pyplot.bar(bin_edges[1:], hist/max(hist), width=width, color='#5B9BD5')
# cdf = numpy.cumsum(hist/sum(hist))
# pyplot.plot(bin_edges[1:], cdf, '-*', color='#ED7D31')
# pyplot.xlim([0, 70])
# pyplot.ylim([0, 1])
# pyplot.grid()
# pyplot.show()

# # 处理数据，未来n_len = 40个位置，有不为0的值，归类为1，否则为归类为0
# time_01 = numpy.array(range(max(raw_values) + 1)) #生成总时间长度的序列，0，1，2，3，4...
raw_values_index = raw_values.reshape(len(raw_values), 1)
all_zeros_index = numpy.zeros(max(raw_values) + 1, int)  # 生成总时间长度的全零序列，量纲1ms
all_zeros_index[raw_values_index] = 1  # 全零序列，在raw_values有值的位置，设1，其余设0
time_index = all_zeros_index
# print(numpy.sum(time_index == 1)) # 有包到达的个数

# # 构造supervised数据集，label是分类结果
sample_num = len(time_index) - (n_row_col + n_len - 1)
n_input = list()
n_output = list()
n_output_block = list()
for ii in range(0, sample_num):
    n_block_1 = time_index[ii:ii + n_row_col] # 顺序截取100*100个时间点，0和1构成的时间序列
    n_block = n_block_1.reshape(n_row, n_col) # 先排行，再排列
    n_input.append(n_block)
    n_label_block_1 = time_index[ii + n_row_col:ii + n_row_col + n_len]
    n_label_block = n_label_block_1.reshape(len(n_label_block_1))
    if (n_label_block == 0).all():
        # n_label_block = numpy.concatenate((n_label_block, [1]), axis = 0)
        n_label = [0, 1]
    else:
        # n_label_block = numpy.concatenate((n_label_block, [0]), axis = 0)
        n_label = [1, 0]
    n_output_block.append(n_label_block)
    n_output.append(n_label)
n_input = numpy.array(n_input)
n_output = numpy.array(n_output)
n_output_block = numpy.array(n_output_block)
# # 统计 sample_num 个样本，未来 n_len ms没有包到达的个数
# n_output_last = n_output[:,-1] # label：1表示没有包到达，否则0
# print(numpy.sum(n_output_last)/sample_num) # 没有包到达占总体的比例，约38% for 20ms

n_input_train = n_input
n_output_train = n_output
n_output_block_train = n_output_block

# fit 模型
lstm_model = fit_lstm(n_input_train, n_output_train, n_col, n_row, output_size, neurons_num, batch_size, epoch_size)  # 训练数据，batch_size，epoche次数, 神经元个数

aa = []