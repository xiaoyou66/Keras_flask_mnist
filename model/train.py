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

# Keras 模型构建主要包括5个步骤：定义（define），编译（compile），训练（fit），评估（evaluate），预测（prediction）。

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


# 下面我们这里就开始构建我们的神经网络，使用Sequential()来构建我们的模型
model = Sequential()
# 第一个卷积层，32个卷积核，大小5x5，卷积模式SAME,激活函数relu,输入张量的大小
# model.add可以给我们的模型添加一层计算模型
# Conv2D 是一个卷积核，会对我们的数据进行卷积计算
# filters表示我们输出空间的维度 会把我们的(28, 28, 1) -> 变成 (28,28,32) 最后面那么filter的大小
# kernel_size表示卷积核的大小
# padding表示是否是否需要padding，Same表示padding完尺寸与原来相同
# activation表示激活函数，一般包括Sigmoid函数、tanh函数、Relu函数等 这个激活函数就是
# 在多层神经网络中，上层节点的输出和下层节点的输入之间具有一个函数关系，这个函数称为激活函数（又称激励函数）。
# input_shape 表示我们的输入的形状
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu',
                 input_shape=(28, 28, 1)))
# 第二次同样是卷积层 输出同样是(None, 28, 28, 32)
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
# 池化层,池化核大小2x2
# pool_size 表示池化窗口大小
# 池化层的输入一般来源于上一个卷积层，主要作用是提供了很强的鲁棒性
# （例如max-pooling是取一小块区域中的最大值，此时若此区域中的其他值略有变化，或者图像稍有平移，pooling后的结果仍不变），
# 并且减少了参数的数量，防止过拟合现象的发生,同时参数的减少对于计算而言也有一定的帮助。
# 而又因为池化层一般没有参数，所以反向传播的时候，只需对输入参数求导，不需要进行权值更新
# 池化后我们的shape变成了 (None, 14, 14, 32)
model.add(MaxPool2D(pool_size=(2, 2)))
# 随机丢弃四分之一的网络连接，防止过拟合
model.add(Dropout(0.25))
# 这我们再创建两个卷积层， 这时我们的输出为 (None, 14, 14, 64)
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
# 这里我们同样对数据进行池化操作 输出为(None, 7, 7, 64)
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
# 全连接层,展开操作 把 (None, 7, 7, 64) 输出为 (None, 3136)
# Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
model.add(Flatten())
# 添加隐藏层神经元的数量和激活函数
# Dense就是常用的全连接层，这个256表示该层的神经单元的结点数
# 这里我们 把上层flatten的数据  (None, 3136)  转换为 (None, 256)
model.add(Dense(256, activation='relu'))
# 我们同样丢弃 1/4 的节点来避免过度拟合
model.add(Dropout(0.25))
# 输出层,这里最终我们，我们这里使用softmax作为激活函数
# 把 (None, 256) -> (None, 10)
# 到这里我们的神经网络就算结束了，注意我们一般中间的隐藏层会使用relu函数，而softmax一般用于输出层
model.add(Dense(10, activation='softmax'))
# summary可以打印出我们的神经网络模型
model.summary()
#  下面这里是我们的神经网络的模型
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 28, 28, 32)        832
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 28, 28, 32)        25632
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
# _________________________________________________________________
# dropout (Dropout)            (None, 14, 14, 32)        0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 14, 14, 64)        18496
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 14, 14, 64)        36928
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 7, 7, 64)          0
# _________________________________________________________________
# flatten (Flatten)            (None, 3136)              0
# _________________________________________________________________
# dense (Dense)                (None, 256)               803072
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 256)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                2570
# =================================================================

# compile用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
# loss表示我们使用的loss函数
# optimizer表示我们使用的优化器名字
# metrics 表示Metrics标注网络评价指标
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit 表示训练模型，这里我们分别输入训练数据，batch_size(每次计算数据量),epochs（训练周期）
# verbose（日志显示，1表示输出进度条记录），validation_data（测试集的输入特征，测试集的标签）
model.fit(x_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
# 评估阶段，这里使用x_test, y_test 这两个来评估我们训练的模型 （评估阶段我们就不打印日志）
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 下面这里就是保存我们的模型了
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
    
model_file = 'model.h5'
model.save(model_file)


# 参考
# 卷积神经网络介绍 https://zhuanlan.zhihu.com/p/42559190
# 卷积神经网络中的filter是怎么工作的 https://blog.csdn.net/qq_21033779/article/details/78211091
# 池化操作 https://blog.csdn.net/mao_xiao_feng/article/details/53453926
# Tensorflow笔记:池化函数 https://blog.csdn.net/FontThrone/article/details/76652762
# Keras中文文档 https://keras-cn.readthedocs.io/en/latest/layers/core_layer/#dense
# tensorflow中model.compile() https://blog.csdn.net/yunfeather/article/details/106461754
# 使用 Keras 手把手介绍神经网络构建  https://yangguang2009.github.io/2016/11/27/deeplearning/develop-neural-network-model-with-keras-step-by-step/

