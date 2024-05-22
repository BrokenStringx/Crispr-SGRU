#
import os
from tensorflow import keras
from keras import Input, Model
from keras.layers import LSTM, Bidirectional, Multiply, BatchNormalization, Concatenate, GRU, Dense, Flatten,Conv2D,Reshape
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.optimizers import Adam
from keras import backend as K
import loss_generator as lg
#

# import os
# import tensorflow as tf
# from tensorflow.keras import Input, Model
# from tensorflow.keras.layers import LSTM, Bidirectional, Multiply, BatchNormalization, Concatenate, GRU, Dense, Flatten,Conv2D,Reshape
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from tensorflow.keras.optimizers import SGD,Adagrad,Adadelta,RMSprop,Nadam
# from tensorflow.keras import backend as K
# import loss_generator as lg

def My_model():
    #inputs = Input(shape=(24, 7))
    inputs = Input(shape=(24, 7), name='main_input')
    inputs_1=Reshape((1,24,7))(inputs)
    conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(inputs_1)
    conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(inputs_1)
    conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(inputs_1)
    conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(inputs_1)
    conv_output = tf.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])
    conv_output = Reshape((24, 40))(conv_output)
    x0 = Bidirectional(GRU(30, return_sequences=True))(conv_output)
    inputs_2=Reshape((24,7))(inputs_1)
    x = Concatenate(axis=2)([inputs_2, x0])

    x1 = Bidirectional(GRU(20, return_sequences=True))(x)
    x = Concatenate(axis=2)([x0, x1])
    
    x2 = Bidirectional(GRU(10, return_sequences=True))(x)
    x = Concatenate(axis=2)([x1, x2])
  
    # x3 = Bidirectional(LSTM(20, return_sequences=True,dropout=0.35))(x)
    # x = Concatenate(axis=2)([x2, x3])

    # x4 = Bidirectional(LSTM(10, return_sequences=True,dropout=0.35))(x)
    
    # x = Concatenate(axis=-1)([x0, x1, x2, x3, x4])
    x = Concatenate(axis=-1)([x0, x1, x2])
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.35)(x)
    x = Dense(2, activation="sigmoid")(x) 

    model = Model(inputs=inputs, outputs=x)
    # print(model.summary())
    opt = Adam(learning_rate=0.0001)
    #opt = SGD(learning_rate=0.0001)
    # model.compile(loss=lg.dice_loss(), optimizer=opt, metrics=['accuracy'])
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
    # model.compile(loss=lg.asymmetric_focal_loss(), optimizer=opt, metrics=['accuracy'])
    return model

