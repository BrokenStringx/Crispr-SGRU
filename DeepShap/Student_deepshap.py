from keras import Input, Model
from keras.layers import LSTM, Bidirectional, Multiply, BatchNormalization, Concatenate, GRU, Dense, Flatten,Conv2D,Reshape
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import os
import random
from keras.utils import np_utils
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization
from tensorflow import keras

from keras import Model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import Encoder_sgRNA_off
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error, roc_curve, precision_recall_curve, auc,matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.callbacks import ReduceLROnPlateau
from datetime import datetime

import shap
shap.initjs()
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

# 定义学生模型
def Student_model():
    Inputs = Input(shape=(24, 16), name='student_input')
    # Inputs= Reshape((24, 16))(Input1)
    x0 = Bidirectional(GRU(30, return_sequences=True))(Inputs)
    x = Concatenate(axis=2)([Inputs, x0])

    x1 = Bidirectional(GRU(20, return_sequences=True))(x)
    x = Concatenate(axis=2)([x0, x1])
    
    x2 = Bidirectional(GRU(10, return_sequences=True))(x)
    x = Concatenate(axis=2)([x1, x2])

    x = Concatenate(axis=-1)([x0, x1, x2])
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.35)(x)
    output = Dense(2, activation="sigmoid")(x)
    student_model = Model(inputs=Inputs, outputs=output)
    opt = Adam(learning_rate=0.0001)
    student_model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
    return student_model


# def load_data(file_path):
#     data_list = []
#     label = []
#     with open(file_path) as f:
#         for line in f:
#             ll = [i for i in line.strip().split(',')]  # strip()表示删除掉数据中的换行符，split（‘，’）则是数据中遇到‘,’ 就隔开。
#             label_item = np.float64(ll[2])
#             data_item = ll[:2]
#             data_list.append(data_item)
#             label.append(label_item)
#     return data_list,label


def load_data(file_path):
    data_list = []
    label = []
    Negative = []
    Positive = []
    with open(file_path) as f:
        for line in f:
            ll = [i for i in line.strip().split(',')]  # strip()表示删除掉数据中的换行符，split（‘，’）则是数据中遇到‘,’ 就隔开。
            label_item = np.float64(ll[2])
            data_item = [int(i) for i in ll[3:]]
            if label_item == 0.0:
                Negative.append(ll[:2])
            else:
                Positive.append(ll[:2])
            data_list.append(data_item)
            label.append(label_item)
    return Negative, Positive, label

def encoding_16(data):
    encode=[]
    for idx, row in data.iterrows():
        on_seq = row[0]
        off_seq = row[1]
        en = Encoder_sgRNA_off.Encoder_16(on_seq=on_seq, off_seq=off_seq, with_category=True)
        encode.append(en.sgRNA_DNA_code)
    return encode



model_name = "student"
data_name = "Dataset"
Negative, Positive, label = load_data(data_name+'.csv')
Positive, Negative = np.array(Positive), np.array(Negative)
num_samples = len(Positive)
random_indices = np.random.choice(len(Negative), num_samples, replace=False)
random_negative = Negative[random_indices]
random_positive = Positive[:num_samples]

print("Positive:", len(random_positive))
print("Negative:", len(random_negative))

Sum=np.vstack((Negative, Positive))
random_indices = np.random.choice(len(random_positive),len(random_positive))
random_positive_1 = Positive[random_indices]
random_negative_1 = Negative[random_indices]
random_train=np.vstack((random_negative_1,random_positive_1))


# # BATCH_SIZE = 256
# # TEST_SIZE = 100

# # AUROC = []
# # PRAUC = []
# # F1_SCORE = []
# # PRECISION = []
# # RECALL = []
# # MCC = []
# # NUM_BATCH = int(len(random_train) / BATCH_SIZE)

# # model=Student_model()                    


# # i=0

student=Student_model()
weighs_path = data_name+'_0.h5'
student.load_weights(weighs_path)

Xtest = np.vstack((random_positive, random_negative))
Seq = [i for i in range(len(Xtest))]  # 列表解析
random.shuffle(Seq)  # 将序列中所有元素随机排序
Xtest = Xtest[Seq]  # 将样本顺序打乱
Xtest=np.array(Xtest)
Xtest=pd.DataFrame(Xtest)
result =Xtest
Xpath=data_name+'_student_Negative_Positive.csv'
result.to_csv(Xpath, index=False)


Xtest_oneHot = encoding_16(Xtest)
Xtest_oneHot=np.array(Xtest_oneHot)
Xtest_oneHot = Xtest_oneHot.astype("float64")
Xtest_oneHot=Xtest_oneHot.reshape((len(Xtest_oneHot),24, 16))


x_train_all = random_train
result =pd.DataFrame(x_train_all)
Xpath=data_name+'_student_Negative_Positive.csv'
result.to_csv(Xpath, index=False)

x_train_all=pd.DataFrame(x_train_all)
x_train_all=encoding_16(x_train_all)
x_train_all=np.array(x_train_all)
e = shap.DeepExplainer(student, x_train_all)
Test=Xtest_oneHot
shap_values = e.shap_values(Test)[1]
print(shap_values.shape)

# shap_values = np.sum(shap_values, axis=-1)
# print(shap_values.shape)

# shap_values = shap_values / 16
shap_values = np.sum(shap_values, axis=0)
shap_values = shap_values / len(Test)
print(shap_values.shape)
shap_values = np.sqrt(np.abs(shap_values))
# print(shap_values)
# shap.image_plot(shap_values,-Test)
shap=pd.DataFrame(shap_values)
shap = shap.transpose() 



    
    