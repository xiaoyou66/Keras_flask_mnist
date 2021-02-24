#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/15 14:27
# @Author  : Paulson
# @File    : app.py
# @Software: PyCharm
# @define  : function

import re
import base64
import numpy as np
import tensorflow.keras as keras
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)

model_file = 'model/model.h5'
# 这里我们使用 keras 来加载我们的模型
global model
model = keras.models.load_model(model_file)

@app.route('/')
def index():
    # 显示主页数据
    return render_template("index.html")  # 如果没有使用 redis 统计访问次数功能，请使用index.html

@app.route('/predict/', methods=['Get', 'POST'])
def preditc():
    # 首先保存我们的图片
    parseImage(request.get_data())
    # 这里我们使用keras提供的函数来把我们的图片数据解析为(28,28) 矩阵
    img = img_to_array(load_img('output.png', target_size=(28, 28), color_mode="grayscale")) / 255.
    # expand_dims 可以进行升维操作， (28,28) -> (1, 28, 28, 1)
    img = np.expand_dims(img, axis=0)
    # 这里我们使用我们的模型来进行预测，因为我们的手写数字就是分类问题，所以使用predict_classes 来获取我们图片的分类
    code = model.predict_classes(img)[0]
    return str(code)


def parseImage(imgData):
    imgStr = re.search(b'base64,(.*)', imgData).group(1)
    # 这里我们把我们手写的数据保存到output.png
    with open('./output.png', 'wb') as output:
        output.write(base64.decodebytes(imgStr))
    


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3335)