#使用CRISPR-net编码
import os
import random
from keras.utils import np_utils
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization
from matplotlib import pyplot as plt
from tensorflow import keras

from keras import Model

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import Encoder_sgRNA_off
import deepshap_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.metrics import mean_squared_error, roc_curve, precision_recall_curve, auc,matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.callbacks import ReduceLROnPlateau

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

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
                Negative.append(ll)
            else:
                Positive.append(ll)
            data_list.append(data_item)
            label.append(label_item)
    return Negative, Positive, label


def train_flow(Train_Negative, Train_Positive, batchsize):
    train_Negative = Train_Negative
    train_Positive = Train_Positive
    Num_Positive = len(train_Positive)
    Num_Negative = len(train_Negative)
    Index_Negative = [i for i in range(Num_Negative)]
    np.random.seed(2020)
    Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')  # batchsize256
    # numpy.random.randint(low, high=None, size=None, dtype='l')返回一个随机整型数，范围从低（包括）到高（不包括）
    # Index_Positive=[0~Num_Positive,共256个]
    random.shuffle(Index_Negative)
    Total_num_batch = int(Num_Negative / batchsize)
    num_counter = 0
    X_input = []
    Y_input = []
    i = 0
    while True:
        i += 1
        np.random.seed(i)
        for i in range(Total_num_batch):
            for j in range(batchsize):
                X_input.append(train_Negative[Index_Negative[j + i * batchsize]])
                Y_input.append(0)
                X_input.append(train_Positive[Index_Positive[j]])
                Y_input.append(1)
                num_counter += 1
                if num_counter == batchsize:
                    Y_input = np_utils.to_categorical(Y_input)  # 把类别标签转换为onehot编码
                    yield (np.array(X_input), np.array(Y_input))
                    X_input = []
                    Y_input = []
                    Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')
                    num_counter = 0


def valid_flow(Test_Negative, Test_Positive, batchsize):
    valid_Negative = Test_Negative
    valid_Positive = Test_Positive
    Num_Positive = len(valid_Positive)
    Num_Negative = len(valid_Negative)
    Index_Negative = [i for i in range(Num_Negative)]

    np.random.seed(2020)
    Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')

    random.shuffle(Index_Negative)
    num_counter = 0
    X_input = []
    Y_input = []
    i = 0
    while True:
        i += 1
        np.random.seed(i)
        for j in range(batchsize):
            X_input.append(valid_Negative[Index_Negative[j]])
            Y_input.append(0)
            X_input.append(valid_Positive[Index_Positive[j]])
            Y_input.append(1)
            num_counter += 1
            if num_counter == batchsize:
                Y_input = np_utils.to_categorical(Y_input)
                yield (np.array(X_input), np.array(Y_input))
                X_input = []
                Y_input = []
                Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')
                num_counter = 0

def encording(data):
    encode=[]
    for idx, row in data.iterrows():
        on_seq = row[0]
        off_seq = row[1]
        label = row[2]
        en = Encoder_sgRNA_off.Encoder(on_seq=on_seq, off_seq=off_seq, with_category=True, label=label)
        encode.append(en.on_off_code)
    return encode

def encordingXtest(Xtest):
    final_code = []
    for idx, row in Xtest.iterrows():
        on_seq = row[0]
        off_seq = row[1]
        en = Encoder_sgRNA_off.Encoder(on_seq=on_seq, off_seq=off_seq, with_category=True, label=label)
        final_code.append(en.on_off_code)
    return final_code


def encordingXtest(Xtest):
    final_code = []
    for idx, row in Xtest.iterrows():
        on_seq = row[0]
        off_seq = row[1]
        en = Encoder_sgRNA_off.Encoder(on_seq=on_seq, off_seq=off_seq, with_category=True, label=label)
        final_code.append(en.on_off_code)
    return final_code


model_name = "Model"
data_name = "Dataset"
Negative, Positive, label = load_data('data_name.csv')
Positive, Negative = np.array(Positive), np.array(Negative)
print("Negative Number:", len(Negative))
num_samples = len(Positive)
random_indices = np.random.choice(len(Negative), num_samples, replace=False)
random_negative = Negative[random_indices]
random_positive = Positive[:num_samples]

print("Positive Number:", len(random_positive))
print("Negative Number:", len(random_negative))

Sum=np.vstack((Negative, Positive))
random_indices = np.random.choice(len(random_positive),len(random_positive))
random_positive_1 = Positive[random_indices]
random_negative_1 = Negative[random_indices]
random_train=np.vstack((random_negative_1,random_positive_1))

Train_Vilidation_Negative, Test_Negative = train_test_split(Negative, test_size=0.15, random_state=42)
Train_Vilidation_Positive, Test_Positive = train_test_split(Positive, test_size=0.15, random_state=42)
Train_Negative, Vilidation_Negative = train_test_split(Train_Vilidation_Negative, test_size=0.2, random_state=42)
Train_Positive, Vilidation_Positive = train_test_split(Train_Vilidation_Positive, test_size=0.2, random_state=42)

Train_Positive = pd.DataFrame(Train_Positive)
Train_Negative = pd.DataFrame(Train_Negative)
Vilidation_Positive = pd.DataFrame(Vilidation_Positive)
Vilidation_Negative = pd.DataFrame(Vilidation_Negative)

Train_Positive_oneHot = []
Train_Negative_oneHot = []
Vilidation_Positive_oneHot = [] 
Vilidation_Negative_oneHot = []
Train_Positive_oneHot=encording(Train_Positive)
Train_Negative_oneHot=encording(Train_Negative)
Vilidation_Positive_oneHot=encording(Vilidation_Positive)
Vilidation_Negative_oneHot=encording(Vilidation_Negative)

Train_Positive_oneHot = np.array(Train_Positive_oneHot)
Train_Negative_oneHot = np.array(Train_Negative_oneHot)
Vilidation_Positive_oneHot = np.array(Vilidation_Positive_oneHot)
Vilidation_Negative_oneHot = np.array(Vilidation_Negative_oneHot)

BATCH_SIZE = 256
TEST_SIZE = 100

AUROC = []
PRAUC = []
F1_SCORE = []
PRECISION = []
RECALL = []
MCC = []
NUM_BATCH = int(len(Train_Negative) / BATCH_SIZE)

model=gru_deepshap.My_model()                    


i=0
Train_Negative_oneHot = Train_Negative_oneHot.astype("float64")
Train_Positive_oneHot = Train_Positive_oneHot.astype("float64")
Vilidation_Negative_oneHot = Vilidation_Negative_oneHot.astype("float64")
Vilidation_Positive_oneHot = Vilidation_Positive_oneHot.astype("float64")
# print(Train_Positive_oneHot.shape)
Train_Positive_oneHot = Train_Positive_oneHot.reshape((len(Train_Positive_oneHot), 24, 7))
Train_Negative_oneHot = Train_Negative_oneHot.reshape((len(Train_Negative_oneHot), 24, 7))
Vilidation_Positive_oneHot = Vilidation_Positive_oneHot.reshape((len(Vilidation_Positive_oneHot), 24, 7))
Vilidation_Negative_oneHot = Vilidation_Negative_oneHot.reshape((len(Vilidation_Negative_oneHot), 24, 7))

weighs_path = "Model.h5"

model.load_weights(weighs_path)

Xtest = np.vstack((random_positive, random_negative))
Seq = [i for i in range(len(Xtest))] 
random.shuffle(Seq)  
Xtest = Xtest[Seq]  
Xtest=np.array(Xtest)
Xtest=pd.DataFrame(Xtest)
result =Xtest
Xpath=data_name+'_Negative_Positive.csv'
result.to_csv(Xpath, index=False)


Xtest_oneHot = encordingXtest(Xtest)
Xtest_oneHot=np.array(Xtest_oneHot)
Xtest_oneHot = Xtest_oneHot.astype("float64")
Xtest_oneHot=Xtest_oneHot.reshape((len(Xtest_oneHot),24, 7))




x_train_all = random_train
result =pd.DataFrame(x_train_all)
Xpath=data_name+'_Negative_Positive.csv'
result.to_csv(Xpath, index=False)

x_train_all=pd.DataFrame(x_train_all)
x_train_all=encording(x_train_all)
x_train_all=np.array(x_train_all)
e = shap.DeepExplainer(model, x_train_all)
Test=Xtest_oneHot
shap_values = e.shap_values(Test)[1]
print(shap_values.shape)

shap_values = np.sum(shap_values, axis=-1)
print(shap_values.shape)

shap_values = shap_values / 7 
shap_values = np.sum(shap_values, axis=0)
shap_values = shap_values / len(Test)
print(shap_values.shape)
shap_values = np.sqrt(np.abs(shap_values))
# print(shap_values)
# shap.image_plot(shap_values,-Test)
shap=pd.DataFrame(shap_values)
shap = shap.transpose() 



    
    