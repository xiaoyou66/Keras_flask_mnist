#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/15 14:14
# @Author  : Paulson
# @File    : train.py
# @Software: PyCharm
# @define  : function
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, MaxPool2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import json

# 注释掉这部分我们就可以关闭GPU加速了
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# 因为我们加载的数据集有5万张图片，训练的时候是不可能一次性都读取的，
# 所以我们可以一次读取128张，这样我们就需要读取469次才能全部读取完
batch_size = 128
# 因为最后计算的结果是一个 [50000,10]的矩阵，这个10用来对y_train 进行升维操作
num_classes = 10
# 一次训练可能不够准确，我们循环训练10次来提高模型的准确率
epochs = 10

# 我们输入的图片的宽度和高度，我们的图片是28*28像素的
img_rows, img_cols = 28, 28

# 加载我们的数据集，我们的数据集包括训练数据和测试数据，他们各自的shape如下
# (60000, 28, 28)  (60000,)  (10000, 28, 28)  (10000,)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 因为彩色图片一般是包括Width， Height， Channels，后面那个是通道，一般的图片是包含RGB这三个通道
# 对于一张 28*28 三通道的图片来说，如果是channels_first 那么组织方式就为 (3,28,28), 否则就是 (28,28,3)
# 下面这里就是进行reshape操作，把我们的三维矩阵变成四维矩阵(60000, 28, 28) ->  (60000, 28, 28, 1)
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 因为训练的时候一般都会把我们的数据缩小范围，我们这里就是把0-255的灰度数据转换为0-1范围，来方便我们操作
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print('x_train shape:', x_train.shape)
# 打印结果如下(60000, 28, 28, 1)

print(x_train.shape[0], 'train samples')
# 60000 train samples （表示有6W个训练样本）
print(x_test.shape[0], 'test samples')
# 10000 test samples（表示有1W个测试样本）

# convert class vectors to binary class matrices
# 因为我们的y_train表示这个图片的标签，初始数据为1维的矩阵
# 刚开始效果如下： [5 0 4 ... 5 6 8]
# 这里相当于升维操作  (60000,) -> (60000,10)
# 最后的结果如下
# [[0. 0. 0. ... 0. 0. 0.]
#  [1. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 1. 0.]]
#  这个很好理解 比如[5] 对应 [[0,0,0,0,0,1,0,0,0,0]]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(input_shape)

# 构建网络
model = Sequential()
# 第一个卷积层，32个卷积核，大小5x5，卷积模式SAME,激活函数relu,输入张量的大小
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
# 池化层,池化核大小２x2
model.add(MaxPool2D(pool_size=(2, 2)))
# 随机丢弃四分之一的网络连接，防止过拟合
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
# 全连接层,展开操作，
model.add(Flatten())
# 添加隐藏层神经元的数量和激活函数
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
# 输出层
model.add(Dense(10, activation='softmax'))
model.summary()
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
#
# with open('model.json', 'w') as outfile:
#     json.dump(model.to_json(), outfile)
#
# model_file = 'model.h5'
# model.save(model_file)
