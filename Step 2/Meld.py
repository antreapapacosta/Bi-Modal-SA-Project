import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import librosa
import librosa.display
import librosa.display
import os
import wavio
import scipy.io.wavfile
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
import pandas as pd
from sklearn.decomposition import KernelPCA

from playsound import playsound
import glob
import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, AveragePooling1D, BatchNormalization, Input, Flatten, Dropout, Activation
from keras.utils import to_categorical, np_utils
import scipy.io.wavfile
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

'''
This script returns the cleaned format of the wav path of MELD utterances
RETURNS:
final_datameld.csv
'''

############ Script to get final_datameld.csv
#os.chdir('C:\\Users\\35799\\OneDrive\\Desktop\\MELD\\wav')
# change directory to user dir
os.chdir('C:\\Users\\Andrea\\Desktop\\Project\\MELD')

# get paths of all wav files in dir
mylist = []
#Train_data_path = 'C:\\Users\\35799\\OneDrive\\Desktop\\MELD\\wav'
Train_data_path = 'C:\\Users\\Andrea\\Desktop\\Project\\MELD\\wav'

for path, subdirs, files in os.walk(Train_data_path):
    for name in files:
        if 'wav' in name:
            mylist.append(os.path.join(path, name))


#directory = "C:\\Users\\35799\\OneDrive\\Desktop\\MELD\\linguistics_meld\\"
directory = "C:\\Users\\Andrea\\Desktop\\Project\\MELD\\"

# read all MELD utterances as a pd dataframe
def read_csv_meld(directory):
    test = pd.read_csv(directory + "test_sent_emo_dya.csv ")
    train = pd.read_csv(directory + "train_sent_emo_dya.csv")
    val = pd.read_csv(directory + "dev_sent_emo_dya.csv")
    return val, train, test
val, train, test = read_csv_meld(directory)
train['Identifier'] = ""
train.is_copy = False

# convert file names to wav paths
for i in range(len(train)):
    
    train['Identifier'][i] = "dia" + str(train['Old_Dialogue_ID'][i])  + "_utt" "" +  str(train['Old_Utterance_ID'][i])  +".wav"

train.drop(columns=['Dialogue_ID','Utterance_ID','Season','Episode'], inplace=True)
train = train.sort_values(by = ['Old_Dialogue_ID','Old_Utterance_ID'], ignore_index = True)
linguistic = train
linguistic.drop(columns=['Old_Dialogue_ID','Old_Utterance_ID','Emotion'], inplace=True)

# add duration of utterances
data_meld =linguistic

s = []
e = []
word_count = []
difference = []
for i in range(len(data_meld)):
    u = dt.strptime(data_meld['StartTime'][i], '%H:%M:%S,%f')
    u_e = dt.strptime(data_meld['EndTime'][i], '%H:%M:%S,%f')
    e.append(u_e.time())
    s.append(u.time())
    word_count.append(len(data_meld['Utterance'][i]))
    difference.append((u_e - u).total_seconds()) 
    #s.append(u.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
data_meld['Duration'] = difference


data_meld.drop(columns=['StartTime','EndTime'], inplace=True)
data_meld['WordCount'] = word_count

# save to selected dir
data_meld.to_csv("C:\\Users\\Andrea\\Desktop\\Project\\\IEMOCAP\\final_datameld.csv")