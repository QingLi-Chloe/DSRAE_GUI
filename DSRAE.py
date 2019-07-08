# RAE
import keras
from keras import backend as K
from keras.utils import Sequence
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives
from keras import regularizers
from sklearn.linear_model import Lasso
from tensorflow.contrib.keras import layers
from sklearn.linear_model import ElasticNet

from numpy import *
import numpy as np
import os
import nibabel as nib
import pandas as pd
from nibabel import cifti2 as ci
from scipy import stats
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import pickle

import sys
import tkinter as Tk
from tkinter import *
import tkinter.font
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation

SUB_NUM = 5
EPOCH_NUM = 2
PATH = "data"
# data collection used in the first time
def RAE_get_data_L():
    global sub_data
    global data
    global len_volume
    global data_path
    global task_name
    task_name = "Language"
    len_volume = 316
    data_path = "/tfMRI_Language_preproc/"
    np.set_printoptions(precision=2, suppress=True)

    dirs = os.listdir(PATH+data_path)

    a = np.memmap('LANGUAGE.mymemmap', dtype='float32', mode='w+', shape=(len_volume * SUB_NUM, 59421))
    del a
    sub = np.memmap('LANGUAGE.mymemmap', dtype='float32', mode='r+', shape=(len_volume * SUB_NUM, 59421))
    sub_data = np.memmap('LANGUAGE.mymemmap', dtype='float32', mode='r+', shape=(len_volume * SUB_NUM, 59421))

    for num1 in range(0, SUB_NUM):
        sub_path = PATH + data_path + dirs[num1] + "/tfMRI_LANGUAGE_LR_Atlas_MSMAll.dtseries.nii"
        img = nib.load(sub_path)
        img = img.get_data()

        img = img[:, 0:59421]
        sub[num1 * len_volume:(num1 + 1) * len_volume, ] = img

## zscore y#####   preprocessing used in the first time

    print(np.shape(sub))
    img_step = len_volume-1  # volumeLength-1
    cnt = 0
    for num2 in range(0, SUB_NUM * (img_step + 1)):
        cnt += 1
        if cnt == (img_step + 1):
            cnt = 0
            sub_data[num2 - img_step: num2 + 1, :] = stats.zscore(sub[num2 - img_step: num2 + 1, :])

    data = np.expand_dims(sub_data, axis=1)
    data = np.reshape(data, (SUB_NUM, len_volume, 59421))

    data = data[0:SUB_NUM, :, :]

def RAE_get_data_W():
    global sub_data
    global data
    global len_volume
    global data_path
    global task_name
    task_name = "WM"
    len_volume = 405
    data_path = "/tfMRI_WM_preproc/"
    np.set_printoptions(precision=2, suppress=True)

    dirs = os.listdir(PATH+data_path)

    a = np.memmap('WM.mymemmap', dtype='float32', mode='w+', shape=(len_volume * SUB_NUM, 59421))
    del a
    sub = np.memmap('WM.mymemmap', dtype='float32', mode='r+', shape=(len_volume * SUB_NUM, 59421))
    sub_data = np.memmap('WM.mymemmap', dtype='float32', mode='r+', shape=(len_volume * SUB_NUM, 59421))

    for num1 in range(0, SUB_NUM):
        sub_path = PATH + data_path + dirs[num1] + "/tfMRI_WM_LR_Atlas_MSMAll.dtseries.nii"
        img = nib.load(sub_path)
        img = img.get_data()

        img = img[:, 0:59421]
        sub[num1 * len_volume:(num1 + 1) * len_volume, ] = img

    ## zscore y#####   preprocessing used in the first time

    print(np.shape(sub))
    img_step = len_volume-1  # volumeLength-1
    cnt = 0
    for num2 in range(0, SUB_NUM * (img_step + 1)):
        cnt += 1
        if cnt == (img_step + 1):
            cnt = 0
            sub_data[num2 - img_step: num2 + 1, :] = stats.zscore(sub[num2 - img_step: num2 + 1, :])

    data = np.expand_dims(sub_data, axis=1)
    data = np.reshape(data, (SUB_NUM, len_volume, 59421))

    data = data[0:SUB_NUM, :, :]

def RAE_get_data_G():
    global sub_data
    global data
    global len_volume
    global data_path
    global task_name
    task_name = "Gambling"
    len_volume = 253
    data_path = "/tfMRI_Gambling_preproc/"
    np.set_printoptions(precision=2, suppress=True)

    dirs = os.listdir(PATH+data_path)

    a = np.memmap('GAMBLING.mymemmap', dtype='float32', mode='w+', shape=(len_volume * SUB_NUM, 59421))
    del a
    sub = np.memmap('GAMBLING.mymemmap', dtype='float32', mode='r+', shape=(len_volume * SUB_NUM, 59421))
    sub_data = np.memmap('GAMBLING.mymemmap', dtype='float32', mode='r+', shape=(len_volume * SUB_NUM, 59421))

    for num1 in range(0, SUB_NUM):
        sub_path = PATH + data_path + dirs[num1] + "/tfMRI_GAMBLING_LR_Atlas_MSMAll.dtseries.nii"
        img = nib.load(sub_path)
        img = img.get_data()

        img = img[:, 0:59421]
        if np.shape(img)[0] == len_volume:
            sub[num1 * len_volume:(num1 + 1) * len_volume, ] = img
        else:
            sub[num1 * len_volume:(num1 + 1) * len_volume - 4, ] = img
            sub[(num1 + 1) * len_volume - 4:(num1 + 1) * len_volume, ] = img[245:250]

    ## zscore y#####   preprocessing used in the first time

    print(np.shape(sub))
    img_step = len_volume - 1  # volumeLength-1
    cnt = 0
    for num2 in range(0, SUB_NUM * (img_step + 1)):
        cnt += 1
        if cnt == (img_step + 1):
            cnt = 0
            sub_data[num2 - img_step: num2 + 1, :] = stats.zscore(sub[num2 - img_step: num2 + 1, :])

    data = np.expand_dims(sub_data, axis=1)
    data = np.reshape(data, (SUB_NUM, len_volume, 59421))

    data = data[0:SUB_NUM, :, :]

#RAE      main code
def RAE_train():

    input_dim = data.shape[-1] # 13
    timesteps =  data.shape[1]# 3

    inputs = Input(shape=(timesteps, input_dim,))
    layer1 = Dense(128, activation='tanh',activity_regularizer=regularizers.l1(1*10e-7), kernel_regularizer=regularizers.l2(1*10e-4))#, activity_regularizer=regularizers.l1(1*10e-7), kernel_regularizer=regularizers.l2(1*10e-4)
    encoded = layer1(inputs)
    encoded = LSTM(64,return_sequences=True,activity_regularizer=regularizers.l1(1*10e-7), kernel_regularizer=regularizers.l2(1*10e-4))(encoded)#
    encoded = LSTM(32,return_sequences=True,activity_regularizer=regularizers.l1(1*10e-7), kernel_regularizer=regularizers.l2(1*10e-4))(encoded)#, activity_regularizer=regularizers.l1(1*10e-7), kernel_regularizer=regularizers.l2(1*10e-4)

    encoder = Model(inputs, encoded)
    encoder.summary()

    inputs = Input(shape=(timesteps, 32,))
    decoded = LSTM(64, return_sequences=True,activation='tanh')(inputs)
    decoded = LSTM(128, return_sequences=True,activation='tanh')(decoded)
    outputs = Dense(input_dim, activation='tanh')(decoded)


    decoder = Model(inputs, outputs)
    decoder.summary()
    # model for RAE
    inputs = Input(shape=(timesteps, input_dim,))
    outputs = encoder(inputs)
    outputs = decoder(outputs)
    sequence_autoencoder = Model(inputs, outputs)


    sequence_autoencoder.compile(optimizer='adam', loss='mse')
    sequence_autoencoder.summary()
    sequence_autoencoder.fit(data, data, epochs=EPOCH_NUM, batch_size=1)

    f_RAE = open('saved_model/DSRAE_'+task_name+'.pickle', 'wb')
    pickle.dump(sequence_autoencoder, f_RAE)
    f_RAE.close()
    f_encoder = open('saved_model/encoder_'+task_name+'.pickle', 'wb')
    pickle.dump(encoder, f_encoder)
    f_encoder.close()
    f_decoder = open('saved_model/decoder_'+task_name+'.pickle', 'wb')
    pickle.dump(decoder, f_decoder)
    f_decoder.close()
#### predict hidden layer ###  calculate y
def RAE_get_y():
    f = open('saved_model/encoder_'+task_name+'.pickle', 'rb')
    encoder = pickle.load(f)
    f.close()
    y = np.zeros((SUB_NUM, len_volume, 32), dtype=float)

    for i in range(0, SUB_NUM):
        y[i * 1: (i + 1) * 1] = encoder.predict(data[i * 1: (i + 1) * 1])

    print(y)

### zscore y#####  and save y

    print(y.shape)
    y = np.reshape(y, (SUB_NUM * len_volume, 1, 32))
    t = range(0, len_volume)
    y_norm = np.zeros((len_volume * SUB_NUM, 1, 32), dtype=float)
    img_step = len_volume-1
    cnt = 0
    for num2 in range(np.shape(y)[0]):
        cnt += 1
        if cnt == (img_step + 1):
            cnt = 0
            y_norm[num2 - img_step: num2 + 1, :] = stats.zscore(y[num2 - img_step: num2 + 1, :])

    where_are_NaNs = isnan(y_norm)
    y_norm[where_are_NaNs] = 0
    sio.savemat(task_name+'_y_norm_test.mat', {'y_norm':y_norm})


###plot task_cope and y_norm#####
def RAE_plot_y():
    global text

    t = range(0, len_volume)
    task = sio.loadmat(PATH+'/'+task_name+'_label.mat')
    task_content = task['Label']

    RAE_plot_y.f.clf()
    RAE_plot_y.a = RAE_plot_y.f.add_subplot(111)

    RAE_plot_y.a.plot(t,task_content[:,1],linewidth=2)

    y_norm = sio.loadmat(task_name+'_y_norm_test.mat')
    y_norm = y_norm['y_norm']

    RAE_plot_y.a.plot(t, y_norm[len_volume*(int(s_sub.get())-1):len_volume*(int(s_sub.get())-0),
                         0,(int(s_net.get())-1)],linewidth=2)

    RAE_plot_y.canvas.draw()

    temp = np.corrcoef(task_content[:,1],y_norm[len_volume*(int(s_sub.get())-1):
                                                len_volume*(int(s_sub.get())-0),0,(int(s_net.get())-1)])
    print('%.3f' % temp[0,1])
    text.config(text="The correlation coefficient is: " + str(temp[0,1]))

##### fit networks by ElasticNet ####
def RAE_get_component():
    y_norm = sio.loadmat(task_name+'_y_norm_test.mat')
    y_norm = y_norm['y_norm']
    y1 = np.squeeze(y_norm)

    clf = ElasticNet(alpha=0.7, l1_ratio=0.005)
    components_img = np.zeros((SUB_NUM, 32, 59421), dtype=float)

    for i in range(0, SUB_NUM):
        a = y1[i * len_volume: (i + 1) * len_volume, :]
        b = sub_data[i * len_volume: (i + 1) * len_volume, :]
        clf.fit(a, b)
        components_img[i, :, :] = np.transpose(clf.coef_)


### average all the subjects' components ###
    components_img_avg = np.zeros((32,59421),dtype = float)
    for i in range(0,59421):
        components_img_avg[:,i] = np.mean(components_img[:,:,i],axis = 0)

### zscore compontents #####  and save
    components_img_avg_norm = np.zeros((32, 59421), dtype=float)

    for num2 in range(0, np.shape(components_img_avg)[0]):
        components_img_avg_norm[num2, :] = stats.zscore(components_img_avg[num2, :])

    where_are_NaNs = isnan(components_img_avg_norm)
    components_img_avg_norm[where_are_NaNs] = 0

    sio.savemat(task_name+'components_img_avg_test.mat', {'components_img_avg_norm':components_img_avg_norm})



#### save patterns####
    image = nib.load(PATH + '/template_for_saving.nii')
    header = image.header
    image_to_write = image.get_data()
    image_to_write[:, :] = 0
    image_to_write[:32, :59421] = components_img_avg_norm

    print(np.shape(image_to_write))

    write_img = ci.Cifti2Image(image_to_write, image.header, image.nifti_header)
    nib.save(write_img, task_name + str(SUB_NUM) + 'sub_LSTM_norm_test.dtseries.nii')

    print('save patterns success!')
#### correlation matrix of networks and volumes ###
# sub_data = np.memmap('sub_LANGUAGE.mymemmap', dtype='float32', mode='r+', shape=(316*791,59421))
def RAE_dec_volume():

    task = sio.loadmat(task_name+'components_img_avg_test.mat')
    components_img_norm = task['components_img_avg_norm']


    corr_net = np.zeros((SUB_NUM, len_volume, 32), dtype=float)
    print(np.shape(corr_net))
    for i in range((s_sub.get()-1), (s_sub.get()-0)):
        a = sub_data[i * len_volume : (i + 1) * len_volume, :]
        for k in range(0, len_volume):
            c = a[k, :]
            for j in range(0, 32):
                b = components_img_norm[j, :]
                temp=np.corrcoef(c,b)
                corr_net[i,k,j] = temp[0,1]#i:subnumber k:timepoint of one sub


### plot the correlation between the original and patterns ###
    RAE_dec_volume.f.clf()
    RAE_dec_volume.b = RAE_dec_volume.f.add_subplot(111)

    a = np.transpose(corr_net[(s_sub.get()-1),:,:])

    im = RAE_dec_volume.b.pcolor(np.arange(0, len_volume), np.arange(0, 32), a,cmap='seismic')

    RAE_dec_volume.canvas.draw()


###### GUI ########

import tkinter
from tkinter import *



window = tkinter.Tk()
window.title('                              DSRAE Analysis GUI')
screenwidth = window.winfo_screenwidth()
screenheight = window.winfo_screenheight()
size = '%dx%d+%d+%d' % (500, 240, (screenwidth - 500) / 2, (screenheight - 240) / 2)
window.geometry(size)
window.maxsize(1600, 1400)
window.minsize(600, 900)
# window.configure(background='#ADD8E6')

ft = tkinter.font.Font(family='Fixdsys', size=10, weight=tkinter.font.BOLD)
ft1 = tkinter.font.Font(size=10)
label1 = Label(window, text = "Data Prepare", font=ft)
label2 = Label(window, text = "DSRAE Analysis Process", font=ft)
label3 = Label(window, text = "Temporal Results", font=ft)
label4 = Label(window, text = "Spatial Results", font=ft)
label5 = Label(window, text = "sub_num", font=ft)
label6 = Label(window, text = "net_num", font=ft)
bt_get_data1 = Button(window,text = "Language", bg = "#A9A9A9", command = RAE_get_data_L, font=ft1, height=2, width=7)
bt_get_data2 = Button(window,text = "WM", bg = "#A9A9A9", command = RAE_get_data_W, font=ft1, height=2, width=7)
bt_get_data3 = Button(window,text = "Gambling", bg = "#A9A9A9", command = RAE_get_data_G, font=ft1, height=2, width=7)


bt_train = Button(window,text = "Analyze Data", bg = "#A9A9A9", command = RAE_train, font=ft1, height=2, width=13)
bt_get_y = Button(window,text = "Get Temporal", bg = "#A9A9A9", command = RAE_get_y, font=ft1, height=2, width=13)
bt_plot_y = Button(window,text = "Plot Temporal", bg = "#A9A9A9", command = RAE_plot_y, font=ft1, height=2, width=13)
bt_get_component = Button(window,text = "Get Spatial", bg = "#A9A9A9", command = RAE_get_component, font=ft1, height=2, width=13)
bt_decode_volime = Button(window,text = "Plot Decoding", bg = "#A9A9A9", command = RAE_dec_volume, font=ft1, height=2, width=13)

text = Label(window, text='', font=ft)

label5.place(x=0, y=210)
s_sub = Scale(window, from_=1, to=SUB_NUM, orient=VERTICAL,
             length=300, showvalue=1, tickinterval=5, resolution=1)
s_sub.place(x=0,y=230)
label6.place(x=530, y=210)
s_net = Scale(window, from_=1, to=32, orient=VERTICAL,
             length=300, showvalue=1, tickinterval=5, resolution=1)
s_net.place(x=530,y=230)

label1.place(x=110,y=30)
bt_get_data1.place(x=40, y=50)
bt_get_data2.place(x=120, y=50)
bt_get_data3.place(x=200, y=50)
label2.place(x=310,y=30,anchor='nw')
bt_train.place(x=330,y=50,anchor='nw')

label3.place(x=100,y=100,anchor='nw')
bt_get_y.place(x=100,y=120,anchor='nw')
bt_plot_y.place(x=100,y=160,anchor='nw')
label4.place(x=340,y=100,anchor='nw')
bt_get_component.place(x=330,y=120,anchor='nw')
bt_decode_volime.place(x=330,y=160,anchor='nw')


RAE_plot_y.f = Figure(figsize=(5,3), dpi=100)
RAE_plot_y.canvas = FigureCanvasTkAgg(RAE_plot_y.f, master=window)
RAE_plot_y.canvas.draw()
RAE_plot_y.canvas.get_tk_widget().place(x=50, y=230)
RAE_dec_volume.f = Figure(figsize=(5,3), dpi=100)
RAE_dec_volume.canvas = FigureCanvasTkAgg(RAE_dec_volume.f, master=window)
RAE_dec_volume.canvas.draw()
RAE_dec_volume.canvas.get_tk_widget().place(x=50, y=570)

text.place(x=50, y=540)

window.mainloop()
