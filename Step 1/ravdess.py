import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import librosa
import librosa.display
import os
import seaborn as sns
import wavio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import IPython.display as ipd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import scipy.io.wavfile
import numpy as np
from pytictoc import TicToc
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import scipy.io.wavfile
import sys
import pandas as pd
import glob
from collections import Counter
import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, AveragePooling1D, BatchNormalization, Input, Flatten, Dropout, Activation
from keras.utils import to_categorical, np_utils
from playsound import playsound
import pandas as pd
import random

# change to directory of ravdess files
os.chdir('C:\\Users\\Andrea\\Desktop\\Dissertation\\trial data\\Audio_Speech_Actors_01-24\\Actor_01')

################################DATA PRE-PROCESSING #########################

def get_emotion(files):
    '''
    Function to parser files using their 
    file name convention.
    Returns: list with final emotion labels
    '''
    final_emotions = []
    for file in files:
        emotion = file[6:8]
        #print(emotion)
        if emotion =='02':
            final_emotions.append('calm')
        elif emotion=='03':
            final_emotions.append('happy')
        elif emotion=='04':
            final_emotions.append('sad')
        elif emotion=='05':
            final_emotions.append('angry')
        elif emotion=='06':
            final_emotions.append('fearful')
        elif emotion=='07':
            final_emotions.append('disgusted')
        elif emotion=='08':
            final_emotions.append('suprised')
        elif emotion=='01':
            final_emotions.append('neutral')
    return final_emotions


def get_dataframe(actor):
    '''
    Function to parse file names and labels from RAVDESS directory,
    for specific actor.
    Actor is the actor number.
    Returns: Data-frame of file names and emotion labels for
    specific actors.
    '''
    path = 'C:\\Users\\Andrea\\Desktop\\Dissertation\\trial data\\Audio_Speech_Actors_01-24\\Actor_' + actor
    files = os.listdir(path)  
    emotions = get_emotion(files)
    data = {'files': files, 'emotions' : emotions}
    df = pd.DataFrame(data = data)
    return df


def get_entire():
    '''
    Function to parse file names and labels from RAVDESS directory,
    for all actors.
    Returns: Data-frame of file names and emotion labels for
    all actors.
    '''
    df = get_dataframe('01')
    for i in range(2,10):
        df2 = get_dataframe('0'+ str(i))
        df = df.append(df2,ignore_index=True)
    for i in range(10,25):
        df2 = get_dataframe(str(i))
        df = df.append(df2, ignore_index=True)
    return df

df = get_entire()


# emotion to sentiment Robinson conversion
my_dict = {'angry': 'negative', 'calm': 'positive', 'disgusted': 'negative',
          'fearful': 'negative', 'happy': 'positive', 'neutral' : 'neutral', 
          'sad': 'negative', 'suprised': 'positive'}
sentiment_m_df = df.replace(my_dict)
Counter(sentiment_m_df['emotions'])

# keep only positive and negative sentiments
sentiment_pos_neg = sentiment_m_df[sentiment_m_df['emotions'].isin(['positive','negative'])].reset_index().iloc[:,1:]


def play_sample(emotion_sample, data_frame, play = False):
    '''
    Function to play audio sample of a specific emotion,
    to develop spectrograms and wave-plots.
    Audio is played only when play = True (default = False).
    Returns: 2 matplotlib.pyplot figures and audio sample 
    '''
    emotions = data_frame['emotions']
    unique = list(set(emotions))
    if emotion_sample not in unique:
        return 'Emotion not recognised, try another emotion.'
    else:
        all_em = list(data_frame[data_frame['emotions'] == emotion_sample]['files'])
        random_emotion = random.choices(all_em, k=1)
        actor = (random_emotion[0][len(random_emotion[0])-6:len(random_emotion[0])-4])
        path = os.getcwd()
        path = path[:len(path)-2] + actor 
        file_path = path + '\\' + random_emotion[0]
        print(file_path)
        if play == True:
            playsound(file_path)
        data, sampling_rate = librosa.load(file_path)
        plot = librosa.display.waveplot(data, sr=sampling_rate)
        plt.title('Emotion : ' + emotion_sample)
        plt.ylabel('Amplitude')
        f_trans = librosa.stft(data)
        Xdb = librosa.amplitude_to_db(abs(f_trans))
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='hz') 
        #If to pring log of frequencies  
        #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar()
        plt.title('Emotion : ' + emotion_sample)
        return plot


play_sample('positive', sentiment_pos_neg, play = True)

def get_file_paths(df):
    '''
    Function to get file paths of all RAVDESS audio utterances,
    Returns: list of all file paths.
    '''
    files = list(df['files'])
    file_paths = []
    path = os.getcwd()
    for file in files:
        actor = (file[len(file)-6:len(file)-4])
        file_path = (path[:len(path)-2] + actor + '\\' + file)
        file_paths.append(file_path)
    return file_paths


all_paths = get_file_paths(df)

# features about silence
def percentage_silence(data, sr, tol):
    '''
    Function to compute percentage silence of an audio file.
    '''
    total_duration = data.shape[0]/sr
    count = data[(abs(0-data)<= tol)].shape[0]/sr
    return count/total_duration * 100
def total_silence_seconds(data, sr, tol):
    '''    
    Function to compute total silence seconds of an audio file.
    '''
    count = data[(abs(0-data)<= tol)].shape[0]/sr
    return count
def max_silent_period(data, sr, tol):
    '''
    Function to compute maximum silent period of an audio file.
    '''
    s_periods = list(abs(0-data)<= tol)
    #print(s_periods)
    count = 1
    initializer = ''
    for period in s_periods:
        if initializer == period and initializer == True:
            count +=1
        else:
            count = 1
        initializer = period
    return count/sr

def create_df(num_coeff, df):
    '''
    Function to initialize empty data-frame with file paths.
    Returns:
    Dataframe with file paths and empty columns.
    '''
    df['perc_silence'] = 0
    df['total_silence_seconds'] = 0
    df['max_silent_period'] = 0
    df['zcrossings'] = 0
    df['tempo'] = 0
    df['spectral_centroids_av'] = 0
    df['spectral_centroids_ch'] = 0
    df['rms'] = 0
    df['gender'] = 0 
    for col in range(1,num_coeff+1):
        name = 'coeff' + str(col)
        df[name] = 0
    for j in range(len(df)):
        if int(df['files'].iloc[j][-6:-4])%2 != 0:
            df.iloc[j,10] = 1
        else:
            df.iloc[j,10] = 0
    return df

df = create_df(30, sentiment_pos_neg)
for j in range(len(df)):
    if int(df['files'].iloc[j][-6:-4])%2 != 0:
        df.iloc[j,10] = 1
    else:
        df.iloc[j,10] = 0

t = TicToc() #create instance of class
def get_features(df, num_coeff):
    '''
    Function to perform feature construction for acoustic features.
    Returns:
    Dataframe with acoustic features.
    '''
    k = 0
    t.tic() #Start timer
    while k <= (len(df)-1):
        for fp in all_paths:
            tol = 1e-3
            data, sr = librosa.load(fp)
            df.iloc[k,2] = percentage_silence(data, sr, tol)
            df.iloc[k,3] = total_silence_seconds(data, sr, tol)
            df.iloc[k,4] = max_silent_period(data, sr, tol)
            df.iloc[k,5] = sum(librosa.zero_crossings(data, pad=False))
            df.iloc[k,6] = librosa.beat.tempo(data)[0]
            spectral_centroids = librosa.feature.spectral_centroid(data, sr=sr)[0]
            df.iloc[k,7] = sum(spectral_centroids)/len(spectral_centroids)
            change_beats = []
            if len(spectral_centroids) != 0:
                for i in range(1,len(spectral_centroids)+1):
                    if i != len(spectral_centroids):
                        change_beats.append(spectral_centroids[i]-spectral_centroids[i-1])

            df.iloc[k,8] = sum(change_beats)/len(change_beats)
            df.iloc[k,9] = sum(np.squeeze(librosa.feature.rms(y=data)))/len(np.squeeze(librosa.feature.rms(y=data)))
            mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=num_coeff).T,axis=0) 
            for col in range(1,num_coeff + 1):
                column_index = 10 + col
                #print(column_index)
                counter = col-1
                #print(df.iloc[1, column_index] )
                df.iloc[k, column_index] = mfccs[counter]
            k += 1
            print(k)
    t.toc() #Time elapsed since t.tic()
    return df

new_df = get_features(df, 30 )

my_dict = {'angry': 'negative', 'calm': 'positive', 'disgusted': 'negative',
          'fearful': 'negative', 'happy': 'positive', 'neutral' : 'neutral', 
          'sad': 'negative', 'suprised': 'positive'}
sentiment_positive_negative_audio = new_df.replace(my_dict)
sent_full = sentiment_positive_negative_audio[sentiment_positive_negative_audio['emotions'].isin(['positive','negative'])].reset_index().iloc[:,1:]
new_df = sent_full
x = new_df.iloc[:,2:]
y =  new_df.iloc[:,1]

################################ANALYSIS#########################
######################### RF Classifier
X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                   test_size=0.30,
                                                   random_state=42)
# define hyperparameter values
n_estimators = [200, 400, 600, 800, 1000, 1500, 2000]
max_depth = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
max_depth.append(None)
max_features = ['auto', 'log2']
criterion = ['entropy', 'gini']
min_samples_split = [2,3,4,5,6,7,8,9,10]
min_impurity_decrease = [0.0, 0.05, 0.1]
bootstrap = [True, False]
# create hyperparameter grid
hyper_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'max_features': max_features,
               'criterion': criterion,
               'min_samples_split': min_samples_split,
               'min_impurity_decrease': min_impurity_decrease,
               'bootstrap': bootstrap}
# initialize random forest classifier
classifier_rf = RandomForestClassifier()

classifier_rf_random = RandomizedSearchCV(estimator = classifier_rf,
                               param_distributions = hyper_grid, 
                               n_iter = 200, cv = 3, verbose = 2,
                               random_state = 42, 
                               n_jobs = -1)
classifier_rf_random.fit(X_train_res, y_train_res)
# view optimal hyper-parameters
classifier_rf_random.best_params_
'''
Returns:
{'n_estimators': 1000,
 'min_samples_split': 8,
 'min_impurity_decrease': 0.0,
 'max_features': 'log2',
 'max_depth': 50,
 'criterion': 'entropy',
 'bootstrap': False}
'''
def cross_validation(k, x, y):
    '''
    Performs k-fold CV of optimized RF model 
    on audio features
    Returns: Average classification metrics
    including: weighted precision and F1-Score,
    accuracy and macro recall
    '''
    result = []
    result_recall = []
    result_precision = []
    result_f1 = []
    actual_result = []
    kf = KFold(n_splits=k, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rf_random_all= RandomForestClassifier(n_estimators = 1000, min_samples_split = 8, min_impurity_decrease = 0.0,
                                max_features = 'log2', max_depth = 50, criterion = 'entropy', bootstrap = False)
        
        rf_random_all.fit(x_train, y_train)
        
        y_pred = rf_random_all.predict(x_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test,y_pred))

        print(accuracy_score(y_test, y_pred))
        result.append(accuracy_score(y_test, y_pred))
        result_recall.append(recall_score(y_test, y_pred, average="macro"))
        result_precision.append(precision_score(y_test, y_pred, average="weighted"))
        result_f1.append(f1_score(y_test, y_pred, average = 'weighted'))
    return sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)

cross_validation(3, x, y)

'''
Returns:
(0.7842261904761906,
 0.7718624948033116,
 0.7841451876964167,
 0.7818201942894237)
'''

#################Examine MFCC and Accuracy

def cross_validation_av(k, x, y):
    '''
    Performs k-fold CV of optimized RF model 
    on audio features
    Returns: Average classification accuracy
    '''
    result = []
    result_recall = []
    result_precision = []
    result_f1 = []
    actual_result = []
    kf = KFold(n_splits=k, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rf_random_all= RandomForestClassifier(n_estimators = 1000, min_samples_split = 8, min_impurity_decrease = 0.0,
                                max_features = 'log2', max_depth = 50, criterion = 'entropy', bootstrap = False)
        
        rf_random_all.fit(x_train, y_train)
        
        y_pred = rf_random_all.predict(x_test)
        print(accuracy_score(y_test, y_pred))
        result.append(accuracy_score(y_test, y_pred))
    return sum(result)/len(result)

# iteratively remove MFCC's and obtain average accuracy
mean_acc = []
for mf_n in range(31):
    print(mf_n)
    if mf_n == 0:
        accuracy = cross_validation_av(3, x, y)
    else:
        accuracy = cross_validation_av(3, x.iloc[:,:-mf_n],y)
    mean_acc.append(accuracy)

# plot of MFCC vs mean accuracy
sns.set(rc={"figure.figsize":(12, 9)}) #width=3, #height=4
sns.set(font_scale = 2, style='whitegrid')
sns.lineplot(data=mfccs_df.iloc[:-1], x="Number of Coefficients", y="Accuracy")

############################## KNN audio model ###################
scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
#List Hyperparameters that we want to tune.
leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
#Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
#Create new KNN object
knn = KNeighborsClassifier()
#Use GridSearch
clf = GridSearchCV(knn, hyperparameters, cv=3, verbose = 10)
#Fit the model
best_model = clf.fit(X_scaled_train,y_train)
#Print The value of best Hyperparameters
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
'''
Returns:
Best leaf_size: 1
Best p: 1
Best n_neighbors: 1
'''
def cross_validation_knn(k, x, y ):
    '''
    Performs k-fold CV of optimized KNN model 
    on audio features
    Returns: Average classification metrics
    including: weighted precision and F1-Score,
    accuracy and macro recall
    '''
    result = []
    result_recall = []
    result_precision = []
    result_f1 = []
    actual_result = []
    kf = KFold(n_splits=k, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        knn_model = KNeighborsClassifier(leaf_size = 1, p = 1, n_neighbors = 1)
        y_train, y_test = y[train_index], y[test_index]
        scaler = preprocessing.StandardScaler().fit(x_train)
        X_scaled_train = scaler.transform(x_train)
        X_scaled_test = scaler.transform(x_test)
        
        knn_model.fit(X_scaled_train, y_train)
        y_pred = knn_model.predict(X_scaled_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test,y_pred))

        print(accuracy_score(y_test, y_pred))
        result.append(accuracy_score(y_test, y_pred))
        result_recall.append(recall_score(y_test, y_pred, average="macro"))
        result_precision.append(precision_score(y_test, y_pred, average="weighted"))
        result_f1.append(f1_score(y_test, y_pred, average="weighted"))
    return sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)

cross_validation_knn(3, x, y )
'''
Returns:
(0.8311011904761906,
 0.8313292686753234,
 0.8332677859020071,
 0.8316024258238505)
'''

######################### LR model
scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)
def cross_validation_lr(k, x, y ):
    '''
    Performs k-fold CV of LR model 
    on audio features
    Returns: Average classification metrics
    including: weighted precision and F1-Score,
    accuracy and macro recall
    '''
    result = []
    result_recall = []
    result_precision = []
    result_f1 = []
    actual_result = []
    kf = KFold(n_splits=k, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        knn_model = KNeighborsClassifier(leaf_size = 1, p = 1, n_neighbors = 1)
        y_train, y_test = y[train_index], y[test_index]
        scaler = preprocessing.StandardScaler().fit(x_train)
        X_scaled_train = scaler.transform(x_train)
        X_scaled_test = scaler.transform(x_test)
        model_logistic = LogisticRegression()
        model_logistic.fit(X_scaled_train, y_train)
        y_pred = model_logistic.predict(X_scaled_test)

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test,y_pred))

        print(accuracy_score(y_test, y_pred))
        result.append(accuracy_score(y_test, y_pred))
        result_recall.append(recall_score(y_test, y_pred, average="macro"))
        result_precision.append(precision_score(y_test, y_pred, average="weighted"))
        result_f1.append(f1_score(y_test, y_pred, average="weighted"))
    return sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)


cross_validation_lr(3, x, y )
'''
Returns:
(0.634672619047619, 0.6209832825822246, 0.6321770598416877, 0.6314205539074867)

'''

###################### NB Classifier
hyper_grid = {
    'var_smoothing': np.logspace(0,-9, num=100)
}
nb_classifier = GridSearchCV(estimator=GaussianNB(), param_grid=hyper_grid,
                            verbose=1, cv=10, n_jobs=-1, scoring = "accuracy")
nb_classifier.fit(X_train, y_train)
print(nb_classifier.best_estimator_)
'''
Returns:
GaussianNB(var_smoothing=0.0002848035868435802)
'''

def cross_validation_nb(k, x, y ):
    '''
    Performs k-fold CV of optimized NB model 
    on audio features
    Returns: Average classification metrics
    including: weighted precision and F1-Score,
    accuracy and macro recall
    '''
    result = []
    result_recall = []
    result_precision = []
    result_f1 = []
    actual_result = []
    kf = KFold(n_splits=k, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        nb_model =GaussianNB(var_smoothing=0.0002848035868435802)
        y_train, y_test = y[train_index], y[test_index]
        nb_model.fit(x_train, y_train)
        y_pred = nb_model.predict(x_test)

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test,y_pred))

        print(accuracy_score(y_test, y_pred))
        result.append(accuracy_score(y_test, y_pred))
        result_recall.append(recall_score(y_test, y_pred, average="weighted"))
        result_precision.append(precision_score(y_test, y_pred, average="weighted"))
        result_f1.append(f1_score(y_test, y_pred, average="weighted"))
    return sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)

cross_validation_nb(3, x, y )
'''
Returns:
 (0.5706845238095238,
 0.5470197703145804,
 0.5596179617415774,
 0.5573601472296844)
 '''


 ##################Human Experiment

# using the play_sample function, the agent was played the following sampled utterances:
l_cor = ['positive','negative', 'negative', 'negative', 
     'negative','positive', 'positive', 'negative',
     'negative','negative', 'positive','negative',
     'positive','positive', 'negative', 'positive',
     'negative','negative', 'negative','positive',
     'positive', 'negative', 'positive','negative',
     'negative', 'negative', 'negative','negative',
     'positive', 'negative', 'negative', 'negative',
     'positive', 'negative', 'negative', 'negative',
     'positive','positive', 'negative', 'negative',
     'negative', 'positive','negative', 'negative',
     'negative', 'negative','negative', 'positive',
     'negative','positive' ]

# the agent gave the following corresponding predictions:
l_m = ['positive','negative', 'negative', 'negative', 
     'negative','positive', 'negative', 'positive',
     'negative','negative', 'negative','negative',
     'negative','negative', 'negative', 'positive',
     'positive','negative', 'negative','negative',
     'positive', 'negative', 'negative','negative',
     'negative', 'negative', 'negative','negative',
     'positive', 'negative', 'negative','negative',
     'negative','negative', 'negative', 'negative',
     'negative','positive', 'negative', 'negative',
     'positive', 'positive','negative', 'negative',
     'negative', 'negative','positive', 'positive',
     'positive', 'negative']

# print their results
print(classification_report(l_cor,l_m))

# plot the grouped bar plot
listb = [83.11, 83.32,83.13, 83.16]
lista = [72, 71,66,71]
# plot the grouped bar plot
raw_data = {
    # cat:    A                  B                  C                    D
    'x': ['Human Classifier','Human Classifier','Human Classifier','Human Classifier',
          'Acoustic KNN Classifier','Acoustic KNN Classifier','Acoustic KNN Classifier','Acoustic KNN Classifier',
         'Acoustic RF Classifier','Acoustic RF Classifier','Acoustic RF Classifier','Acoustic RF Classifier'],
    'y': [72, 71,66,71,
          83.11, 83.32,83.13, 83.16,
         78.42,78.41,77.19,78.18],
    'category': ['Accuracy', 'Weighted Precision', 'Macro Recall', 'Weighted F1 Score',
                'Accuracy', 'Weighted Precision', 'Macro Recall', 'Weighted F1 Score',
                'Accuracy', 'Weighted Precision', 'Macro Recall', 'Weighted F1 Score']
           }
sns.set(font_scale = 2, style="whitegrid",rc={"figure.figsize":(14, 10)})

sns.barplot(x='x', y='y', hue='category', data=raw_data)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
