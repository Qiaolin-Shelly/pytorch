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

train_size = 480000
train_val_size = 384000
val_train_size = 96000
test_size = 350000
batch_size = 512
epoch_size = 1000
neurons_num = 20

parameter_num = 1
difference_order = 1

n_row = 100
n_col = 100
n_len = 40
output_size = 2
n_row_col = n_row * n_col


def create_model(n_col, n_row, output_size, n_neurons, batch_size):
    model = Sequential()
    model.add(Bidirectional(LSTM(n_neurons, dropout=0.2, recurrent_dropout=0.2, input_shape=(n_row, n_col), stateful=False, return_sequences=True)))
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
    # 重塑训练数据格式 [samples, timesteps, features]
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
    model.load_weights('D:/AI_model/Save model/binary_model_07')
    history_loss_list = list()
    history_valloss_list = list()
    history_accuracy_list = list()
    history_valaccuracy_list = list()
    for i in range(0, 1000):
        # # 手动shuffle sample, 否则在 validation split rate 时，顺序截取，而非随机
        # index_permt, X_epoch, y_epoch, X_train, y_train, X_val, y_val = list(), list(), list(), list(), list(), list(), list()
        # index_permt = numpy.random.permutation(index)
        # history = model.fit(X, y, epochs=1, batch_size=n_batch, verbose=2, shuffle=True, validation_split=0.2)  # validation_data = (X_val, y_val))
        history = model.fit(X_train, y_train, epochs=1, batch_size=n_batch, verbose=1, shuffle=True, validation_data=(X_val, y_val))  # validation_split=0.2)
        # history = model.fit((X[index_permt])[0:train_val_size], (y[index_permt])[0:train_val_size], epochs=1,
        #                     batch_size=n_batch, verbose=1, shuffle=True, validation_data=((X[index_permt])[train_val_size:], (y[index_permt])[train_val_size:]))  # validation_split=0.2)
        history_loss_list.append(history.history['loss'])
        history_valloss_list.append(history.history['val_loss'])
        history_accuracy_list.append(history.history['accuracy'])
        history_valaccuracy_list.append(history.history['val_accuracy'])
        # del index_permt, X_epoch, y_epoch, X_train, y_train, X_val, y_val
        # gc.collect()
        # print(history.history['loss'], history.history['val_loss'])
        # model.test_on_batch(X, y, sample_weight=None, reset_metrics=False, return_dict=True)
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
        # # 画出loss，val_loss
        # fig, ax = pyplot.subplots(1, 1)
        # pyplot.plot(history_loss_list)
        # pyplot.plot(history_valloss_list)
        # fig, ax = pyplot.subplots(1, 1)
        # pyplot.plot(history_accuracy_list)
        # pyplot.plot(history_valaccuracy_list)
        model.reset_states()
    model.save_weights('D:/AI_model/Save model/binary_model_08')
    return model


# 将时间序列转换为监督类型的数据序列
def timeseries_to_supervised(data, n_in, n_out, dropnan):  # dropnan=True
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
    for i in range(0, len(yhat)):
        value = yhat[i] + history[-interval - (n_seq - i - 1)]
        y_hat.append(value)
    yhat = numpy.array(y_hat)
    return yhat


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


# 预测
def forcast_lstm(model, X):
    # X = X.reshape(1, 1, len(X))
    # X2 = numpy.concatenate([X]*100, axis = 0)
    yhat = model.predict(X)
    # print((yhat_1 == yhat).all())
    # yhat = yhat[0,:]
    return yhat


# 加载数据
series = read_csv('D:/AI_model/data_set/UDP_original_onlytime_phy.csv', header=0, parse_dates=[0], index_col=0,
                  squeeze=True)
# 处理多个表头相同的csv级联
# series_01 = read_csv('dy_dataset/dy_01.csv', header=0, parse_dates=[0], index_col=0, squeeze=True) # TCP_ceil_del
# series_02 = read_csv('dy_dataset/dy_02.csv', header=0, parse_dates=[0], index_col=0, squeeze=True) # TCP_ceil_del
# frames = [series_01, series_02]
# result = pandas.concat(frames)

# 1ms刻度, 0ms为起始点：考虑到达时间差平均20ms左右，dataset 过大,只能截取一段数据
raw_values = series.values
raw_values = raw_values * 1000
raw_values = raw_values[2500:34000]
raw_values = raw_values - raw_values[0] * numpy.ones(len(raw_values))
raw_values = numpy.array(raw_values, int)

diff_values = difference(raw_values, difference_order)  # 转换成 1-order 差分数据
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

# pyplot.subplot(221)
# pyplot.plot(diff_values[600:799], c = 'b')
# pyplot.subplot(222)
# pyplot.plot(diff_values[800:999], c = 'r')
# pyplot.show()

# # 绘制 差分的PDF histogram
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
############################ 处理数据，ms有值，处理为1，否则为0
# time_01 = numpy.array(range(max(raw_values) + 1)) #生成总时间长度的序列，0，1，2，3，4...
raw_values_index = raw_values.reshape(len(raw_values), 1)
all_zeros_index = numpy.zeros(max(raw_values) + 1, int)  # 生成总时间长度的全零序列，量纲1ms
all_zeros_index[raw_values_index] = 1  # 全零序列，在raw_values有值的位置，设1，其余设0
time_index = all_zeros_index
# numpy.sum(time_index == 1) # 有包到达的个数，实际等于raw_values的长度
############################ 构造supervised数据集，label是分类结果
sample_num = len(time_index) - (n_row_col + n_len - 1)
n_input = list()
n_output = list()
n_output_block = list()
for ii in range(0, sample_num):
    n_block_1 = time_index[ii:ii + n_row_col]
    n_block = n_block_1.reshape(n_row, n_col)
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

n_input_train = n_input[0:train_size]
n_output_train = n_output[0:train_size]
n_output_block_train = n_output_block[0:train_size]

time_index_test = time_index[train_size:train_size + test_size + (n_row_col + n_len - 1)]
# time_index_test = time_index[train_size - (n_row_col + n_len - 1) - test_size:train_size]
n_input_test = n_input[train_size:(train_size + test_size)]
n_output_test = n_output[train_size:(train_size + test_size)]
n_output_block_test = n_output_block[train_size:(train_size + test_size)]

# predictions = numpy.load("D:/AI_model/Results/predictions_binary_model_01.npy")
# numpy.where(predictions >= 0, predictions, 1)
# numpy.where(predictions < 0, predictions, 0)

# fit 模型
lstm_model = fit_lstm(n_input_train, n_output_train, n_col, n_row, output_size, neurons_num, batch_size, epoch_size)  # 训练数据，batch_size，epoche次数, 神经元个数

# # 用于解决LSTM模型的batch_size和用于test时，batch_size = 1不匹配的问题, 提出这种解决方法的真是个大聪明，无语
# # LSTM模型设置的 batch_input_shape 主要是要和stateful关联使用，stateful-LSTM，sample之间的状态有记忆性，batch_0的sample_0，会把状态传递给batch_1的sample_0
# # 如果不复制 dropout， recurrent_dropout，
# new_model = Sequential()
# new_model.add(LSTM(neurons_num, input_shape=(n_row, n_col), stateful=True, return_sequences=True))
# new_model.add(LSTM(neurons_num, input_shape=(n_row, n_col), stateful=True))
# # new_model.add(Dropout(0.1))
# new_model.add(Dense(output_size))
# old_weights = lstm_model.get_weights()
# new_model.set_weights(old_weights)
# new_model.compile(loss='binary_crossentropy', optimizer='adam') # categorical_crossentropy

# 如果复制 dropout， recurrent_dropout，直接creat model
new_model = create_model(n_col, n_row, output_size, neurons_num, batch_size)
new_model.load_weights('D:/AI_model/Save model/binary_model_08')

# 测试数据的前向验证，实验发现，如果训练次数很少的话，模型会简单的把数据后移; 训练次数过多，会过拟合，性能反而变差很多，滞后问题也没解决！
predictions_y_hat = list()
predictions_yhat = list()
expected = list()
num_pre_true = 0
num_pre_false = 0
num_all_0 = 0
num_all_1 = 0
num_pre_true = 0
num_pre_false = 0
miss_n_len_num = 0

miss_pdc_num = 0
dec_pdc_num = 0
save_pdc_num = 0
# Initial AI_model input
pointer = 0
n_input_test_ii = time_index_test[pointer:pointer + n_row_col]
n_input_test_ii = n_input_test_ii.reshape(n_row, n_col)
n_input_test_ii = n_input_test_ii.reshape(1, n_row, n_col)
n_input_test_ii = tensorflow.cast(n_input_test_ii, tensorflow.bfloat16)
n_input_test_ii = numpy.array(n_input_test_ii)

n_output_block_test_ii = time_index_test[pointer + n_row_col:pointer + n_row_col + n_len]
if (n_output_block_test_ii == 0).all():
    n_output_test_ii = [0, 1]
else:
    n_output_test_ii = [1, 0]

win_pre = [[0 for i in range(2)] for j in range(test_size + 100000)]
win_pre[0] = [0, n_row_col]
win_num = 1 # 预测窗的个数

sleep_win = [[0 for i in range(2)] for j in range(test_size + 100000)]
sleep_win_num = 0 # sleep的个数
fig, ax = pyplot.subplots(1, 1)
prediction_num = 0
# n_input_test_ii = n_input_test[0]
# n_input_test_ii = n_input_test_ii.reshape(1, n_input_test_ii.shape[0], n_input_test_ii.shape[1])
# n_input_test_ii = tensorflow.cast(n_input_test_ii, tensorflow.bfloat16)
# n_input_test_ii = numpy.array(n_input_test_ii)
while pointer < test_size:
    prediction_num = prediction_num + 1
    y_real = n_output_test_ii
    y_real_block = n_output_block_test_ii
    # if (y_real[0:n_len] == 0).all():
    #     num_all_0 = num_all_0 + 1
    # else:
    #     num_all_1 = num_all_1 + 1
    if y_real[0] == 0:
        num_all_0 = num_all_0 + 1  # [0 1] 没有到达
    else:
        num_all_1 = num_all_1 + 1  # [1 0] 有到达
    # print(num_all_1) # 统计有到达的个数

    yhat = forcast_lstm(new_model, n_input_test_ii)
    predictions_yhat.append(yhat)

    if yhat[0][0] >= yhat[0][1]:
        y_hat = [1, 0]  # 预测 - 有包
    else:
        y_hat = [0, 1]  # 预测 - 没有包

    if y_hat[0] == 1:  # 预测 - 有包
        if y_real[0] == 1:  # 真实 - 有包
            y_real_eq_1 = numpy.where(n_output_block_test_ii == 1)
            dec_pdc_num = dec_pdc_num + y_real_eq_1[0][0] + 1
            win_pre[win_num] = [pointer + n_row_col, pointer + n_row_col + y_real_eq_1[0][0]]
            ax.add_patch(
                pch.Rectangle(
                    (win_pre[win_num][0], 1),  # (x,y)
                    (win_pre[win_num][1] - win_pre[win_num][0]),  # width
                    0.0003,  # height
                    edgecolor='black',
                    facecolor='green',
                    fill=True
                )
            )
            win_num = win_num + 1
            pointer = pointer + y_real_eq_1[0][0] + 1
        else:  # 真实 - 无包
            # n_input_test_ii_2 = n_input_test_ii.reshape(n_row_col)
            # t1 = n_input_test_ii_2.tolist()
            # t2 = numpy.zeros(n_len, int)
            # t2 = t2.tolist()
            # t3 = t1 + t2
            # t4 = t3[n_len:] # t4 = t3[-n_row_col:]
            # n_input_test_ii = t4.reshape(1, n_row, n_col)
            dec_pdc_num = dec_pdc_num + n_len
            win_pre[win_num] = [pointer + n_row_col, pointer + n_row_col + n_len - 1]
            ax.add_patch(
                pch.Rectangle(
                    (win_pre[win_num][0], 1),  # (x,y)
                    (win_pre[win_num][1] - win_pre[win_num][0]),  # width
                    0.0003,  # height
                    edgecolor='black',
                    facecolor='green',
                    fill=True
                )
            )
            win_num = win_num + 1
            pointer = pointer + n_len
    else:  # 预测 - 无包
        sleep_win[sleep_win_num] = [pointer + n_row_col,  pointer + n_row_col + n_len - 1]
        ax.add_patch(
            pch.Rectangle(
                (sleep_win[sleep_win_num][0], 0.99999),  # (x,y)
                (sleep_win[sleep_win_num][1] - sleep_win[sleep_win_num][0]),  # width
                0.00001,  # height
                edgecolor='black',
                facecolor='pink',
                fill=True
            )
        )
        sleep_win_num = sleep_win_num + 1
        if y_real[0] == 1:  # 真实 - 有包
            miss_pdc_num = miss_pdc_num + numpy.sum(n_output_block_test_ii == 1)
            save_pdc_num = save_pdc_num + n_len
            miss_n_len_num = miss_n_len_num + 1
            pointer = pointer + n_len
        else:  # 真实 - 无包
            save_pdc_num = save_pdc_num + n_len
            pointer = pointer + n_len

    n_input_test_ii = time_index_test[pointer:pointer + n_row_col]
    n_input_test_ii = n_input_test_ii.reshape(n_row, n_col)
    n_input_test_ii = n_input_test_ii.reshape(1, n_row, n_col)
    n_input_test_ii = tensorflow.cast(n_input_test_ii, tensorflow.bfloat16)
    n_input_test_ii = numpy.array(n_input_test_ii)

    n_output_block_test_ii = time_index_test[pointer + n_row_col:pointer + n_row_col + n_len]
    if (n_output_block_test_ii == 0).all():
        n_output_test_ii = [0, 1]
    else:
        n_output_test_ii = [1, 0]

    predictions_y_hat.append(y_hat)
    # y_hat = numpy.array(y_hat)
    # if (y_hat[0] == y_real[0]) & (y_hat[1] == y_real[1]):
    #     num_pre_true = num_pre_true + 1
    # else:
    #     num_pre_false = num_pre_false + 1
    #     if (y_hat[0] == 0) & (y_real[0] == 1):
    #         miss_pdc_num = miss_pdc_num + sum(y_real_block)
    # print('Index=%d' % (i + 1), num_pre_true, num_pre_false, miss_pdc_num)
    # print("Miss", miss_pdc_num)
    # print("Decoded", dec_pdc_num)
    # print("Save", save_pdc_num)
    print('Index=%d, Miss=%d, Decoded=%d, Save=%d' % (prediction_num, miss_pdc_num, dec_pdc_num, save_pdc_num))

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

# predictions = numpy.array(predictions)
# numpy.save('D:/AI_model/Results/label_binary_model_02', n_output_test)
# numpy.save('D:/AI_model/Results/predictions_binary_model_02', predictions)
# predictions = numpy.load("D:/AI_model/Results/predictions_binary_model_01.npy")
# label = numpy.load("D:/AI_model/Results/label_binary_model_01.npy")
# miss_num = 0
# for i in range(0, test_size):
#     if (label[i][0] == 1) & (predictions[i][0] == 0):
#         miss_num = miss_num + 1
#
# arrive_num = 0
# for i in range(0, test_size):
#     if (label[i][0] == 1):
#         arrive_num = arrive_num + 1
time_real = numpy.where(time_index_test == 1)
time_real = time_real[0]
pyplot.scatter(time_real, 1*numpy.ones(len(time_real)), s=10, c='r')
# pyplot.xlim(3599, 23800)
pyplot.ylim((0.9998, 1.0002))

# numpy.save('D:/AI_model/Results/win_pre', win_pre)
# numpy.save('D:/AI_model/Results/sleep_win', sleep_win)
raw_in_win_current_index = []
raw_in_win_current_index = numpy.array(raw_in_win_current_index)
for ii in range(0, sleep_win_num):
    # sleep_win_ii = numpy.linspace(sleep_win[ii][0], sleep_win[ii][1], 40).astype(int)
    # sleep_win_ii = numpy.linspace(sleep_win[ii][0] + 5, sleep_win[ii][1] - 5, 30).astype(int)
    sleep_win_ii = numpy.linspace(sleep_win[ii][0] + 10, sleep_win[ii][1] - 10, 20).astype(int)
    sleep_win_ii = sleep_win_ii.reshape(10, 2)
    sleep_win_ii = sleep_win_ii[:, 0]
    kk = 0
    for iii in range(0, len(sleep_win_ii)):
        kk = numpy.where(time_real == sleep_win_ii[iii])
        raw_in_win_current_index = numpy.append(raw_in_win_current_index, kk)
aa = []