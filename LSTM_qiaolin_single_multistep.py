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
warnings.filterwarnings("ignore", category=numpy.VisibleDeprecationWarning)

# 读取时间数据的格式化
#def parser(x):
#   return x

batch_size = 100
# time_step_num = 1
train_size = 100000
epoch_size = 1000
neurons_num = 40
# memory_value = 1
parameter_num = 1
n_lag = 10 # 利用过去n_lag个观测，预测未来n_seq个值
n_seq = 1
difference_order = 1

# # 转换成有监督数据，输入 + 输出
# def timeseries_to_supervised(data, n_lag, n_seq):
#     df = DataFrame(data)
#     columns = [df.shift(i) for i in range(n_lag, 0, -1)]  # 数据滑动一格，作为input，df原数据为output
#     columns.append(df)
#     df = concat(columns, axis=1)
#     df.fillna(0, inplace=True)
#     return df

# 将时间序列转换为监督类型的数据序列
def timeseries_to_supervised(data, n_in, n_out, dropnan): # dropnan=True
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # 这个for循环是用来输入列标题的 var1(t-1)，var1(t)，var1(t+1)，var1(t+2)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # 转换为监督型数据的预测序列 每四个一组，对应 var1(t-1)，var1(t)，var1(t+1)，var1(t+2)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # 拼接数据
    agg = concat(cols, axis=1)
    agg.columns = names
    # 把null值转换为0
    if dropnan:
        agg.dropna(inplace=True)
    print(agg)
    return agg

# 转换成差分数据，一阶差分
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
    for i in range(0,len(yhat)):
        value = yhat[i] + history[-interval - (n_seq - i - 1)]
        y_hat.append(value)
    yhat = numpy.array(y_hat)
    return yhat
    # ############ 2阶-差分预测
    # return yhat + history[-interval] + history[-interval] - history[-interval - 1]
    # ############ 3阶-差分预测
    # return yhat + history[-interval] + history[-interval] + history[-interval] - history[-interval - 1] - history[-interval - 1] - history[-interval - 1] + history[-interval - 2]
    # ############ 4阶-差分预测
    # return yhat + 4 * history[-interval] - 6 * history[-interval - 1] + 4 * history[-interval - 2] + history[-interval - 3]
    # ############ 5阶-差分预测
    # return yhat + 5 * history[-interval] - 10 * history[-interval - 1] + 10 * history[-interval - 2] - 5 * history[-interval - 3] + history[-interval - 4]

# 缩放
def scale(train, test):
    # 根据训练数据建立缩放器
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = numpy.array(train)
    train = train.reshape(len(train), 1)
    test = numpy.array(test)
    test = test.reshape(len(test), 1)
    scaler = scaler.fit(train)
    # 转换train data
    train_scaled = scaler.transform(train)
    # 转换test data
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# 逆缩放
def invert_scale(scaler, X, value):
    # new_row = [x for x in X] + [value]
    # array = numpy.array(new_row)
    # array = array.reshape(1, len(array))
    value = value.reshape(len(value), 1)
    inverted = scaler.inverse_transform(value)
    inverted = inverted.reshape(1, len(inverted))
    return inverted

# 匹配LSTM网络训练数据
def fit_lstm(train_input, train_output, n_lag, n_seq, n_batch, nb_epoch, n_neurons, time_step_num): # 观测size: n_lag 预测size: n_seq
    # 重塑训练数据格式 [samples, timesteps, features]
    X = train_input
    y = train_output
    y = y.reshape(y.shape[0], y.shape[1])
    ####################### 手动shuffle sample, 否则在 validation split rate 时，顺序截取，而非随机
    index = [i for i in range(len(X))]
    numpy.random.shuffle(index)
    X = X[index]
    y = y[index]
    ##### BiLSTM需要如下转换
    X = tensorflow.cast(X, tensorflow.float32)
    y = tensorflow.cast(y, tensorflow.float32)
    # 配置一个LSTM神经网络，添加网络参数
    model = Sequential()
    # model.add(LSTM(n_neurons, input_shape=(X.shape[1], X.shape[2]), stateful=False, return_sequences=True))
    # # model.add(LSTM(n_neurons, batch_input_shape=(1024, X.shape[1], X.shape[2]), stateful=False, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(n_neurons, batch_input_shape=(1024, X.shape[1], X.shape[2]), stateful=False, return_sequences=True))
    # model.add(Dropout(0.3))
    # model.add(BatchNormalization())
    # model.add(LSTM(n_neurons, input_shape=(X.shape[1], X.shape[2]), stateful=False, return_sequences=False))
    model.add(Bidirectional(LSTM(n_neurons, input_shape=(X.shape[1], X.shape[2]), stateful=False, return_sequences=False)))
    # # model.add(LSTM(n_neurons, input_shape=(X.shape[1], X.shape[2]), stateful=False, return_sequences=False))
    # model.add(Dropout(0.1))
    # # model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=False, return_sequences=True, dropout=0.3,i kernel_regularizer=regularzers.L1L2(l1=1e-5, l2=1e-5)))
    # # model.add(BatchNormalization())
    # # model.add(Dropout(0.5))
    # model.add(LSTM(150, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=False))
    # model.add(Dropout(0.1))
    # # model.add(BatchNormalization())
    # model.add(Dense(y.shape[1], activation='softmax'))
    model.add(Dense(y.shape[1]))
    # model.add(Activation(keras.activations.softmax))
    # model.add(BatchNormalization())
    # rmsprop = optimizers.RMSprop(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy']) #categorical_crossentropy

    # 调用网络，迭代数据对神经网络进行训练，最后输出训练好的网络模型
    history_loss_list = list()
    history_valloss_list = list()
    history_accuracy_list = list()
    history_valaccuracy_list = list()

    for i in range(nb_epoch):
        model.reset_states()
        history = model.fit(X, y, epochs=1, batch_size=n_batch, verbose=2, shuffle=True, validation_split=0.4)
        history_loss_list.append(history.history['loss'])
        history_valloss_list.append(history.history['val_loss'])
        history_accuracy_list.append(history.history['accuracy'])
        history_valaccuracy_list.append(history.history['val_accuracy'])
        # print(history.history['loss'], history.history['val_loss'])
        # rmse = model.test_on_batch(X, y, sample_weight=None, reset_metrics=False, return_dict=True)
        # rmse_value = list(rmse.values())[0]
        # rmse_list.append(rmse_value)
        # print("当前计算次数loss：" + str(i), history.history['loss'], history.history['val_loss'])
        # print("当前计算次数accuracy："+str(i), history.history['accuracy'], history.history['val_accuracy'])
        # yyhat = model.predict(X, 100)
        # # yyhat = list()
        # # for i in range(0, len(train)):  # 根据测试数据进行预测，取测试数据的一个数值作为输入，计算出下一个预测值，以此类推
        # #     XX, yy = train[i, 0:n_lag], train[i, n_lag:]
        # #     yy_hat = forcast_lstm(model, 100, XX)
        # #     yyhat.append(yy_hat)
        # # yyhat = numpy.array(yyhat)
        # rmse_value = sqrt(mean_squared_error(train_label, yyhat))
        # rmse_list.append(rmse_value)
        # ########## 画出loss，val_loss
        # fig, ax = pyplot.subplots(1, 1)
        # pyplot.plot(history_loss_list)
        # pyplot.plot(history_valloss_list)
        # fig, ax = pyplot.subplots(1, 1)
        # pyplot.plot(history_accuracy_list)
        # pyplot.plot(history_valaccuracy_list)
    model.save_weights('D:/AI_model/binary_model_01')
    return model

# 1步长预测
def forcast_lstm(model, X):
    # X = X.reshape(1, 1, len(X))
    # X2 = numpy.concatenate([X]*100, axis = 0)
    yhat = model.predict(X)
    # print((yhat_1 == yhat).all())
    # yhat = yhat[0,:]
    return yhat

# 加载数据
# series = read_csv('dy_dataset/dy_time_partial.csv', header=0, parse_dates=[0], index_col=0, squeeze=True) # TCP_ceil_del
series = read_csv('data_set/UDP_original_onlytime_phy_partial.csv', header=0, parse_dates=[0], index_col=0, squeeze=True) # TCP_ceil_del

# series_01 = read_csv('dy_dataset/dy_01.csv', header=0, parse_dates=[0], index_col=0, squeeze=True) # TCP_ceil_del
# series_02 = read_csv('dy_dataset/dy_02.csv', header=0, parse_dates=[0], index_col=0, squeeze=True) # TCP_ceil_del
# series_03 = read_csv('dy_dataset/dy_03.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_04 = read_csv('dy_dataset/dy_04.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_05 = read_csv('dy_dataset/dy_05.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_06 = read_csv('dy_dataset/dy_06.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_07 = read_csv('dy_dataset/dy_07.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_08 = read_csv('dy_dataset/dy_08.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_09 = read_csv('dy_dataset/dy_09.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_10 = read_csv('dy_dataset/dy_10.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_11 = read_csv('dy_dataset/dy_11.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_12 = read_csv('dy_dataset/dy_12.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_13 = read_csv('dy_dataset/dy_13.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_14 = read_csv('dy_dataset/dy_14.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_15 = read_csv('dy_dataset/dy_15.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_16 = read_csv('dy_dataset/dy_16.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_17 = read_csv('dy_dataset/dy_17.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_18 = read_csv('dy_dataset/dy_18.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_19 = read_csv('dy_dataset/dy_19.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_20 = read_csv('dy_dataset/dy_20.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_21 = read_csv('dy_dataset/dy_21.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_22 = read_csv('dy_dataset/dy_22.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_23 = read_csv('dy_dataset/dy_23.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_24 = read_csv('dy_dataset/dy_24.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# series_25 = read_csv('dy_dataset/dy_25.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# frames = [series_01, series_02, series_03, series_04, series_05, series_06, series_07, series_08, series_09, series_10, series_11, series_12, series_13, series_14, series_15, series_16, series_17, series_18,
#           series_19, series_20, series_21, series_22, series_23, series_24, series_25]
# result = pandas.concat(frames)

# 让数据变成稳定的
raw_values = series.values
raw_values = raw_values * 1000
raw_values = numpy.array(raw_values, int)
raw_values = raw_values[0:5000]
raw_values = raw_values -  raw_values[0]*numpy.ones(len(raw_values))
raw_values = numpy.array(raw_values, int)
########### 去掉有0元素的行：一阶差分 len(raw_values) - 1，是差分后所有数据的个数；前（n_lag + n_seq - 1）行，有0元素；去掉有0元素的行，之后划分为train + test
# test_size = len(raw_values) - difference_order - (n_lag + n_seq - 1) - train_size
diff_values = difference(raw_values, difference_order) # 转换成 1-order 差分数据
## 处理为非线性数据，目的是隐藏数据的时序特征，防止出现滞后现象，但是发现这个非线性处理没有个锤子用
# diff_values_a = numpy.array(diff_values)
# diff_values = numpy.sqrt(diff_values_a)
# 预测2阶差分，举例记忆窗，memory_value = 4: { t2-t1-(t1-t0) t3-t2-(t2-t1) t4-t3-(t3-t2) t5-t4-(t4-t3) } 预测 t6-t5-(t5-t4)
# diff_values_2 = difference(diff_values, 1) # 转换成 2-order 差分数据
# # ###### diff_values_3: 预测3阶差分，举例记忆窗，memory_value = 4:
# # { (t2-t1)-(t1-t0) (t3-t2)-(t2-t1) (t4-t3)-(t3-t2) (t5-t4)-(t4-t3) } 预测 t6-t5-(t5-t4)
# 预测3阶差分
# diff_values_3 = difference(diff_values_2, 1) # 转换成 3-order 差分数据
# diff_values = diff_values_3
# diff_values = numpy.array(diff_values)
# pyplot.plot(diff_values)
# pyplot.show()
# pyplot.subplot(221)
# pyplot.plot(diff_values[600:799], c = 'b')
# pyplot.subplot(222)
# pyplot.plot(diff_values[800:999], c = 'r')
# pyplot.show()
# ####### 不差分，直接对原始时序序列，做预测。实验证明，效果特别差，可能是程序需要修改
# diff_values = raw_values
# # 绘制 二阶查分的PDF histogram ####################################################
# pyplot.subplot(221)
# pyplot.hist(diff_values_2)
# # 获得 histogram 数据
# pyplot.subplot(222)
# hist, bin_edges = numpy.histogram(diff_values_2)
# pyplot.plot(hist)
# # 拟合 fit histogram curve
# pyplot.subplot(223)
# sns.distplot(diff_values_2, kde=False, fit=stats.gamma, rug=True)
# pyplot.show()
# # 同时绘制 归一化的 PDF + CDF       ###############################################
# hist, bin_edges = numpy.histogram(diff_values_2)
# width = (bin_edges[1] - bin_edges[0]) * 0.8
# pyplot.bar(bin_edges[1:], hist/max(hist), width=width, color='#5B9BD5')
# cdf = numpy.cumsum(hist/sum(hist))
# pyplot.plot(bin_edges[1:], cdf, '-*', color='#ED7D31')
# pyplot.xlim([-2, 2])
# pyplot.ylim([0, 1])
# pyplot.grid()
# pyplot.show()
# #################################################################################
# 数据缩放
# 对原始 1st-order 差分数据，scale, scale原则是用train的参数，scale test的数据
diff_values = diff_values.values
diff_values = numpy.array(diff_values, int)
# fig, ax = pyplot.subplots(1, 1)
# pyplot.plot(diff_values)
# ecdf = sm.distributions.ECDF(diff_values)
# fig, ax = pyplot.subplots(1, 1)
# x_x = numpy.linspace(min(diff_values), max(diff_values), 100000)
# y_y = ecdf(x_x)
# pyplot.step(x_x, y_y)
# print(numpy.sum(diff_values > 40)/5000)
############################ 处理数据，ms有值，处理为1，否则为0
# time_01 = numpy.array(range(max(raw_values) + 1)) #生成总时间长度的序列，0，1，2，3，4...
raw_values_index = raw_values.reshape(len(raw_values), 1)
all_zeros_index = numpy.zeros(max(raw_values) + 1, int) #生成总时间长度的全零序列，量纲1ms
all_zeros_index[raw_values_index] = 1 #全零序列，在raw_values有值的位置，设1，其余设0
time_index = all_zeros_index
# numpy.sum(time_index == 1) # 有包到达的个数，实际等于raw_values的长度
############################ 构造supervised数据集，label是分类结果
time_step_num = 10
n_row = time_step_num
n_col = 10
n_len = 40
n_row_col = n_row*n_col
sample_num = len(time_index) - (n_row_col + n_len - 1)
n_input = list()
n_output = list()
for ii in range(0, sample_num):
    n_block_1 = time_index[ii:ii+n_row_col]
    n_block = n_block_1.reshape(n_row, n_col)
    n_input.append(n_block)
    n_label_block_1 = time_index[ii+n_row_col:ii+n_row_col+n_len]
    n_label_block = n_label_block_1.reshape(len(n_label_block_1))
    if (n_label_block == 0).all():
        # n_label_block = numpy.concatenate((n_label_block, [1]), axis = 0)
        n_label_block = [0]
    else:
        # n_label_block = numpy.concatenate((n_label_block, [0]), axis = 0)
        n_label_block = [1]
    # n_label_block = n_label_block.reshape(n_len + 1, 1)
    # n_label_block = n_label_block.reshape(1, 1)
    n_output.append(n_label_block)
n_input = numpy.array(n_input)
n_output = numpy.array(n_output)
# ################################### 统计 sample_num 个样本，未来10个没有包到达的个数
# n_output_last = n_output[:,-1] #label：1表示没有包到达，否则0
# print(numpy.sum(n_output_last)/sample_num) #有包到达占总体的比例，月65.73%
test_size = 10000
n_input_train = n_input[0:train_size]
n_output_train = n_output[0:train_size]
n_input_test = n_input[train_size:(train_size+test_size)]
n_output_test = n_output[train_size:(train_size+test_size)]
# difference_train = diff_values[0:-test_size]
# difference_test = diff_values[-test_size:]
# scaler, difference_train_scaled, difference_test_scaled = scale(difference_train, difference_test)
# diff_values_scale = numpy.append(difference_train_scaled, difference_test_scaled)
# diff_values_scale = diff_values_scale.reshape(len(diff_values_scale), 1)
# supervised = timeseries_to_supervised(diff_values_scale, n_lag, n_seq, 1) # 最后一个元素是记忆性
# supervised_values = supervised.values
# # supervised_values = supervised_values[:, :-1]
#
# # # 标签组 前memory_value因为没凑齐历史观测数据，所以用于训练不准，discard；修改后的程序，已经在timeseries_to_supervised函数中去掉有0元素的标签，无需重复设置
# # supervised_values = supervised_values[(n_lag + n_seq - 1):, :]
#
# # 数据拆分：训练数据、测试数据，前24行是训练集，后12行是测试集
# train_scaled, test_scaled = supervised_values[0:-test_size], supervised_values[-(test_size + time_step_num - 1):]
#
# # train_tt = train.reshape(220, 1, 900)
# # train_tt = train_tt[: :2]
# # tt3 = train_tt.reshape(11000,1,9)
# # tt3 = tt3.reshape(11000,9)
# # train = tt3
# # # 数据缩放
# # scaler, train_scaled, test_scaled = scale(train, test)

# fit 模型
lstm_model = fit_lstm(n_input_train, n_output_train, n_col, n_len, batch_size, epoch_size, neurons_num, n_row)  # 训练数据，batch_size，epoche次数, 神经元个数
#################### 在训练集上验证预测效果
# train_reshaped = train_scaled[:, 0:-1].reshape(len(train_scaled), 1, parameter_num*n_lag)#训练数据集转换为可输入的矩阵 # 最后一个元素是记忆性（如果是单一元素预测），否则是记忆性*元素个数
# train_reshaped = train_scaled[:,:-1].reshape(len(train_scaled), 1, parameter_num*n_lag)#训练数据集转换为可输入的矩阵 # 最后一个元素是记忆性（如果是单一元素预测），否则是记忆性*元素个数
# test_reshaped = test_scaled[:,:-1].reshape(len(test_scaled), 1, parameter_num*n_lag)#训练数据集转换为可输入的矩阵 # 最后一个元素是记忆性（如果是单一元素预测），否则是记忆性*元素个数
# lstm_model.predict(train_reshaped, batch_size = batch_size)#用模型对训练数据矩阵进行预测


########################### 用于解决LSTM模型的batch_size和用于test时，batch_size = 1不匹配的问题。提出这种解决方法的真是个大聪明，无语!!!
####### LSTM模型设置的 batch_input_shape 主要是要和stateful关联使用，stateful-LSTM，sample之间的状态有记忆性，batch_0的sample_0，会把状态传递给batch_1的sample_0
# batch_size1 = 1
# new_model = Sequential()
# # 添加LSTM层
# # new_model.add(LSTM(neurons_num, batch_input_shape=(batch_size1, 1, parameter_num*memory_value), stateful=True)) # X.shape[1], X.shape[2]
# new_model.add(LSTM(neurons_num, batch_input_shape=(batch_size1, 1, parameter_num*memory_value), stateful=True, return_sequences=True)) # X.shape[1], X.shape[2]
# # new_model.add(Dropout(0.1))
# new_model.add(LSTM(neurons_num, batch_input_shape=(batch_size1, 1, parameter_num*memory_value), stateful=True))
# # new_model.add(Dropout(0.1))
# new_model.add(Dense(1))  # 输出层1个node
# old_weights = lstm_model.get_weights()
# new_model.set_weights(old_weights)
# new_model.compile(loss='mean_squared_error', optimizer='adam')


# 测试数据的前向验证，实验发现，如果训练次数很少的话，模型会简单的把数据后移，以昨天的数据作为今天的预测值，当训练次数足够多的时候, 才会体现出来训练结果。
# 实验发现训练次数过多，会过拟合，性能反而变差很多，滞后问题也没解决！
predictions = list()
expected = list()
num_pre_true = 0
num_pre_false = 0
num_all_0 = 0
num_all_1 = 0
num_pre_true = 0
num_pre_false = 0
for i in range(0, test_size):#根据测试数据进行预测，取测试数据的一个数值作为输入，计算出下一个预测值，以此类推
    # 1步长预测
    y_real = n_output_test[i]
    if (y_real[0:n_len] == 0).all():
        num_all_0 = num_all_0 + 1
    else:
        num_all_1 = num_all_1 + 1
    # if (y_real[0] == 0).all():
    #     num_all_0 = num_all_0 + 1
    # else:
    #     num_all_1 = num_all_1 + 1
    print(num_all_1)
    n_input_test_ii = n_input_test[i]
    n_input_test_ii = n_input_test_ii.reshape(1, n_input_test_ii.shape[0], n_input_test_ii.shape[1])
    yhat = forcast_lstm(lstm_model, n_input_test_ii)
    yhat = yhat.reshape(yhat.shape[1], yhat.shape[0])
    # yhat = yhat.reshape(yhat.shape[0])
    # 逆缩放
    # yhat = invert_scale(scaler, X, yhat)
    # 逆差分
    # yhat = yhat.reshape(len(yhat))
    # yhat = numpy.square(yhat)
    # yhat = numpy.log(yhat)
    # # 直接预测原始序列，不作差分，无需inverse
    # yhat = yhat
    predictions.append(yhat)
    # predictions.append(yhat[0])
    # 1阶-差分 预测 ######################################################################
    # expected_value_list = list()
    # for ii in range(0, n_seq):
    #     expected_value = raw_values[len(train_scaled) + i + (n_lag + n_seq - 1) + 1 - (n_seq - ii - 1)]
    #     expected_value_list.append(expected_value)
    # expected_value_list_toarray = numpy.array(expected_value_list)
    # expected.append(expected_value_list_toarray)
    # # # 2阶-差分 预测 ######################################################################
    # # expected = raw_values[len(train) + i + memory_value + 2]
    # # # 3阶-差分 预测 ######################################################################
    # # expected = raw_values[len(train) + i + memory_value + 3]
    # # # 无差分，直接对序列做预测
    # # expected = raw_values[len(train) + i + memory_value + 0]
    # # expected = raw_values[len(train) + i + memory_value][0]
    # # print('Moth=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))
    # print('Index=%d' % (i + 1), numpy.where(yhat == max(yhat)), numpy.where(y_real == max(y_real)))

    index_yhat = numpy.where(yhat == max(yhat))
    index_yhat = index_yhat[0][0]
    index_real = numpy.where(y_real == max(y_real))
    index_real = index_real[0][0]
    if index_real == index_yhat:
        num_pre_true = num_pre_true + 1
    else:
        num_pre_false = num_pre_false + 1
    print('Index=%d' % (i + 1), num_pre_true, num_pre_false)
    # print('Moth=%d, Predicted=%f, Expected=%f' % (i + 1, yhat[0], expected))

# 性能报告
# predictions = numpy.array(predictions)
# expected = numpy.array(expected)
# for i in range(0, n_seq):
#     rmse = sqrt(mean_squared_error(expected[:,i], predictions[:,i]))
#     print('Index=%d' % (i + 1), rmse)
# fig, ax = pyplot.subplots(1, 1)
# pyplot.plot(predictions)
# pyplot.plot(expected)
# fig, ax = pyplot.subplots(1, 1)
# pyplot.plot(expected - predictions)