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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.naive_bayes import MultinomialNB
from playsound import playsound
import glob
import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import keras
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, AveragePooling1D, BatchNormalization, Input, Flatten, Dropout, Activation
from keras.utils import to_categorical, np_utils
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import librosa
import librosa.display
import librosa.display
import os
import wavio
import scipy.io.wavfile
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import sys
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from playsound import playsound
import glob
import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import keras
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, AveragePooling1D, BatchNormalization, Input, Flatten, Dropout, Activation
from keras.utils import to_categorical, np_utils

# read linguistic features of combined data-set
df_ling_bow = pd.read_csv('C:\\Users\\Andrea\\Desktop\\Project\\IEMOCAP\\' + 'bow.csv').iloc[:,1:]

df_ling = pd.read_csv('C:\\Users\\Andrea\\Desktop\\Project\\IEMOCAP\\tfidf_ch2.csv').iloc[:,1:]

df = df_ling.drop_duplicates()
df_n = df[df['Sentiment'].isin(['positive','negative'])].reset_index().iloc[:,1:]

to_keep = set(df_n['Identifiers'])

audio = pd.read_csv('audio_iemocap.csv').iloc[:,1:]

#df = trail_1.drop('wav_path', axis = 1)


# read IEMOCAP audio features
df_audio = pd.read_csv('C:\\Users\\Andrea\\Desktop\\Project\\IEMOCAP\\audio_iemocap.csv').iloc[:,1:]
for row in range(len(df_audio)):
    print(df_audio['wav_path'][row].split('\\')[-1][:-4])
    df_audio['wav_path'][row] = df_audio['wav_path'][row].split('\\')[-1][:-4]

df_audio.columns = ['Identifier', 'Sentiment', 'WordCount', 'perc_silence',
       'total_silence_seconds', 'max_silent_period', 'zcrossings', 'tempo',
       'spectral_centroids_av', 'spectral_centroids_ch', 'rms', 'coeff1',
       'coeff2', 'coeff3', 'coeff4', 'coeff5', 'coeff6', 'coeff7', 'coeff8',
       'coeff9', 'coeff10', 'coeff11', 'coeff12', 'coeff13', 'coeff14',
       'coeff15', 'coeff16', 'coeff17', 'coeff18', 'coeff19', 'coeff20',
       'coeff21', 'coeff22', 'coeff23', 'coeff24', 'coeff25', 'coeff26',
       'coeff27', 'coeff28', 'coeff29', 'coeff30']

# read MELD audio features
audio_meld = pd.read_csv('C:\\Users\\Andrea\\Desktop\\Project\\MELD\\audio_meld_c.csv').iloc[:,2:].drop('Duration', axis = 1)
audio_classifier_df = pd.concat([audio_meld, df_audio], ignore_index = True)
new_df_audio = audio_classifier_df[audio_classifier_df['Identifier'].isin(to_keep)].drop_duplicates().reset_index().iloc[:,1:]
new_to_keep = set(new_df_audio['Identifier'])

# define tf-idf linguistic features for Combined data-set
new_df_ling = df_n[df_n['Identifiers'].isin(new_to_keep)].drop_duplicates().reset_index().iloc[:,1:]

# fix labels in all set of features
correct_sentiments = new_df_ling[['Sentiment', 'Identifiers']]
correct_sentiments.columns = ['Sentiment', 'Identifier']
correct_sentiments[correct_sentiments['Identifier'].str.contains('dia')].to_csv('attempted3.csv')


attempted_audio = new_df_audio.drop('Sentiment',axis=1)
# define audio features for Combined data-set
correct_audio = pd.merge(correct_sentiments, attempted_audio, how='inner', on = 'Identifier')
# construct bow linguistic features for Combined data-set
new_df_ling_bow = df_ling_bow[df_ling_bow['Identifiers'].isin(new_to_keep)].reset_index().iloc[:,1:]

## define features and labels for acoustic, linguistic TF-IDF and BoW data-sets
x = correct_audio.drop('Identifier',axis=1).drop('Sentiment', axis =1 )
y = correct_audio.drop('Identifier',axis=1)['Sentiment']

xl = new_df_ling.drop('Identifiers',axis=1).drop('Sentiment', axis =1 )
yl = new_df_ling.drop('Identifiers',axis=1)['Sentiment']

xl_b = new_df_ling_bow.drop('Identifiers',axis=1).drop('Sentiment', axis =1 )
yl_b = new_df_ling_bow.drop('Identifiers',axis=1)['Sentiment']

####################################Acoustic Models###################
###########Random Forest Tuning
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

n_estimators = [200, 400, 600, 800, 1000, 1500, 2000]
max_depth = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
max_depth.append(None)
max_features = ['auto', 'log2']
criterion = ['entropy', 'gini']
min_samples_split = [2,3,4,5,6,7,8,9,10]
min_impurity_decrease = [0.0, 0.05, 0.1]
bootstrap = [True, False]
hyper_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'max_features': max_features,
               'criterion': criterion,
               'min_samples_split': min_samples_split,
               'min_impurity_decrease': min_impurity_decrease,
               'bootstrap': bootstrap}
classifier_rf = RandomForestClassifier()
classifier_rf_random = RandomizedSearchCV(estimator = classifier_rf,
                               param_distributions = hyper_grid, 
                               n_iter = 200, cv = 3, verbose = 10,
                               random_state = 42, 
                               n_jobs = -1)
classifier_rf_random.fit(X_train, y_train)
# view optimal hyper-parameters
classifier_rf_random.best_params_
'''
Returns: 
{'n_estimators': 2000,
'min_samples_split': 4,
'min_impurity_decrease': 0.0,
'max_features': 'auto',
'max_depth': 15,
'criterion': 'gini',
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
        rf_random_all= RandomForestClassifier(n_estimators=2000, min_samples_split=4,
                                 min_impurity_decrease=0.0, max_features = 'auto',max_depth = 15, 
                                 criterion = 'gini', bootstrap = False)
        
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
(0.6600660066006601,
0.5582233147120155,
0.6366154557888054,
0.6047549019562685)
'''

#KNN training
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
Best p: 2
Best n_neighbors: 14
'''''
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
        knn_model = KNeighborsClassifier(leaf_size = 1, p = 2, n_neighbors = 14)
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
(0.6439526305571733,
0.5422600993672683,
0.6083882034278284,
0.5886335301257096)
'''

#LR training
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
(0.6348281887012232,
0.5296720157730208,
0.5918039330179886,
0.5733200695612872)
'''

#NB training

hyper_grid = {
    'var_smoothing': np.logspace(0,-9, num=100)
}
nb_classifier = GridSearchCV(estimator=GaussianNB(), param_grid=hyper_grid,
                            verbose=1, cv=10, n_jobs=-1, scoring = "accuracy")
nb_classifier.fit(X_train, y_train)
print(nb_classifier.best_estimator_)

'''
Returns:
GaussianNB(var_smoothing=1.0)
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
        nb_model =GaussianNB(var_smoothing=1)
        y_train, y_test = y[train_index], y[test_index]
        nb_model.fit(x_train, y_train)
        y_pred = nb_model.predict(x_test)

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test,y_pred))

        print(accuracy_score(y_test, y_pred))
        result.append(accuracy_score(y_test, y_pred))
        result_recall.append(recall_score(y_test, y_pred, average="macro"))
        result_precision.append(precision_score(y_test, y_pred, average="weighted"))
        result_f1.append(f1_score(y_test, y_pred, average="weighted"))
    return sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)

cross_validation_nb(3, x, y )
'''
Returns:
(0.6431760823141137,
0.5007564990914456,
0.6600783937129052,
0.5051860746399611)
 '''
###################################################LINGUISTIC MODELS#######################################
# RF ling
X_train, X_test, y_train, y_test = train_test_split(xl, yl, test_size=0.30)
n_estimators = [200, 400, 600, 800, 1000, 1500, 2000]
max_depth = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
max_depth.append(None)
max_features = ['auto', 'log2']
criterion = ['entropy', 'gini']
min_samples_split = [2,3,4,5,6,7,8,9,10]
min_impurity_decrease = [0.0, 0.05, 0.1]
bootstrap = [True, False]
hyper_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'max_features': max_features,
               'criterion': criterion,
               'min_samples_split': min_samples_split,
               'min_impurity_decrease': min_impurity_decrease,
               'bootstrap': bootstrap}
classifier_rf = RandomForestClassifier()
classifier_rf_random = RandomizedSearchCV(estimator = classifier_rf,
                               param_distributions = hyper_grid, 
                               n_iter = 200, cv = 3, verbose = 10,
                               random_state = 42, 
                               n_jobs = -1)
classifier_rf_random.fit(X_train, y_train)
# view optimal hyper-parameters
classifier_rf_random.best_params_
'''
Returns:
{'n_estimators': 600,
'min_samples_split': 4,
'min_impurity_decrease': 0.0,
'max_features': 'sqrt',
'max_depth': None,
'criterion': 'entropy',
'bootstrap': False}
'''
def cross_validation_ling(k, x, y):
    '''
    Performs k-fold CV of optimized RF model 
    on linguistic features
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
        rf_random_ling = RandomForestClassifier(n_estimators=600, min_samples_split=4,
                                 min_impurity_decrease=0.0, max_features = 'sqrt',max_depth = None, 
                                 criterion = 'entropy', bootstrap = False)
        
        rf_random_ling.fit(x_train, y_train)
        
        y_pred = rf_random_ling.predict(x_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test,y_pred))

        print(accuracy_score(y_test, y_pred))
        result.append(accuracy_score(y_test, y_pred))
        result_recall.append(recall_score(y_test, y_pred, average="macro"))
        result_precision.append(precision_score(y_test, y_pred, average="weighted"))
        result_f1.append(f1_score(y_test, y_pred, average = 'weighted'))
    return sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)

cross_validation_ling(3, xl, yl)
'''
Returns:
(0.7460687245195107,
0.687057862883265, 
0.7435071146280348,
0.7313116306903468)
'''

cross_validation_ling(3, xl_b, yl_b)
'''
Returns:
(0.7445156280333917,
0.6831838988540323,
0.7403455285267436,
0.7289401589108288)
'''
# KNN ling
X_train, X_test, y_train, y_test = train_test_split(xl, yl, test_size=0.30)
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
Best leaf_size: 3
Best p: 2
Best n_neighbors: 3
'''
def cross_validation_knn_ling(k, x, y, p, leaf_size, n_neighbours ):
    '''
    Performs k-fold CV of optimized KNN model 
    on linguistic features
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
        knn_model = KNeighborsClassifier(leaf_size = leaf_size, p = p, n_neighbors = n_neighbours)
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

cross_validation_knn_ling(3, xl, yl, p = 2, leaf_size = 3, n_neighbours = 3 )
'''
Returns:
(0.7004465152397593, 0.6271527288236896, 0.688931595734942, 0.6763669109473662)
'''
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(xl_b, yl_b, test_size=0.30)
scaler = preprocessing.StandardScaler().fit(X_train_b)
X_scaled_train_b = scaler.transform(X_train_b)
X_scaled_test_b = scaler.transform(X_test_b)
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
best_model = clf.fit(X_scaled_train_b,y_train_b)
#Print The value of best Hyperparameters
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
'''
Returns:
Best leaf_size: 39
Best p: 2
Best n_neighbors: 3
'''
cross_validation_knn_ling(3, xl_b, yl_b, p = 2 , leaf_size = 39, n_neighbours = 3 )
'''
Returns:
(0.6950106775383421,
0.6257940413695227,
0.6815154557366364,
0.6738881066187448)
'''

# LR model

def cross_validation_lr_ling(k, x, y ):
    '''
    Performs k-fold CV of LR model 
    on linguistic features
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
        knn_model = LR(leaf_size = 1, p = 1, n_neighbors = 1)
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

cross_validation_lr_ling(3,xl,yl)

'''
Returns:
(0.7018054746651137,
0.6699112433813784,
0.6988818649901964,
0.7001201908362219)
'''
cross_validation_lr_ling(3,xl_b,yl_b)

'''
Returns:
(0.6985051446321102,
0.6668766786334315,
0.6960086980987135,
0.6968901359263414)
'''

def cross_validation_nb_ling(k, x, y ):
    '''
    Performs k-fold CV of NB model 
    on linguistic features
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
        nb_model =MultinomialNB(alpha=1, fit_prior=True)
        y_train, y_test = y[train_index], y[test_index]
        nb_model.fit(x_train, y_train)
        y_pred = nb_model.predict(x_test)

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test,y_pred))

        print(accuracy_score(y_test, y_pred))
        result.append(accuracy_score(y_test, y_pred))
        result_recall.append(recall_score(y_test, y_pred, average="macro"))
        result_precision.append(precision_score(y_test, y_pred, average="weighted"))
        result_f1.append(f1_score(y_test, y_pred, average="weighted"))
    return sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)

cross_validation_nb_ling(3, xl, yl)
'''
Returns:
(0.705105804698117, 0.6194907343133528, 0.701003925666218, 0.67040344385373)
'''
cross_validation_nb_ling(3, xl_b, yl_b)
'''
Returns:
(0.7297612114152591,
0.6549023549059431,
0.7289043633490607,
0.7048475176428983)
'''

 ##############################################FUSION MODELS##############################################
# Checking explained variance for PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=400)
pr_component = pca.fit_transform(for_pca)
pca_df = pd.DataFrame(data = pr_component)
pca_df_final = pca_df.add_prefix('PrincipalComponent')
pca_df_final
# obtain the total variance explained by pca component
print(round(sum(list(pca.explained_variance_ratio_))*100, 2))
# Returns: 54.51
correct_audio_pca = correct_audio.iloc[:,3:]

pca = PCA(n_components=1)
pr_components = pca.fit_transform(x)
principal_df = pd.DataFrame(data = pr_components)
#pca.explained_variance_ratio_
principalDffinal_audio = principal_df.add_prefix('PrincipalComponentAudio')
print(round(sum(list(pca.explained_variance_ratio_))*100, 2))
# Returns: 99.59

new_df = pd.concat([new_df_ling.iloc[:,:-3].iloc[:,:-3],new_df_ling['Sentiment'], principalDffinal_audio], axis = 1)
new_df_all = pd.concat([new_df_ling.iloc[:,:-3],new_df_ling['Sentiment'], correct_audio_pca], axis = 1)
# define feature fusion features with and without pca
y_ff = new_df['Sentiment']
x_ff = new_df.drop('Sentiment',axis = 1)
y_ff_all = new_df_all['Sentiment']
x_ff_all = new_df_all.drop('Sentiment',axis = 1)


def cross_validation_all(k, x, y):
    '''
    Performs k-fold CV of Feature Fusion model 
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
        rf_random_ling = RandomForestClassifier(n_estimators=600, min_samples_split=4,
                                 min_impurity_decrease=0.0, max_features = 'sqrt',max_depth = None, 
                                 criterion = 'entropy', bootstrap = False)
        
        rf_random_ling.fit(x_train, y_train)
        
        y_pred = rf_random_ling.predict(x_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test,y_pred))

        print(accuracy_score(y_test, y_pred))
        print(f1_score(y_test, y_pred, average = 'weighted'))
        result.append(accuracy_score(y_test, y_pred))
        result_recall.append(recall_score(y_test, y_pred, average="macro"))
        result_precision.append(precision_score(y_test, y_pred, average="weighted"))
        result_f1.append(f1_score(y_test, y_pred, average = 'weighted'))
    return sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)

cross_validation_all(3, x_ff, y_ff)
# PCA FF
'''
Returns:
(0.7474276839448651,
0.6912035540057175,
0.7424448538885198,
0.7345217458453441)

'''
 # NON PCA FF
cross_validation_all(3, x_ff_all, y_ff_all)

'''
Returns:
(0.706658901184236, 
0.6050998626279428, 
0.7259006729208671, 
0.6546594744061505)
 
'''

 # ALL AROUND FUSION

def all_around_fusion_all_s_2(X_trainff, X_testff,
                        y_trainff, y_testff,
                        X_trainl, X_testl,
                        y_trainl, y_testl,
                        X_train, X_test,
                        y_train, y_test,
                        rfaud, rfling, rfall):   
    '''Performs All-Around Fusion.
    Returns: classifications for test data-set.''' 
    total_preds = []
    y_pred_all = rfall.predict_proba(X_testff)
    index = 0
    for preds in list(y_pred_all):
        #print(preds)
        #print(preds)
        actual_pred = int(np.where(preds == max(preds))[0][0])
        if actual_pred == 1:
            total_preds.append('positive')
        else:
            X_test_now_ling = X_testl.iloc[index]
            y_pred_ling = rfling.predict_proba(np.array(X_test_now_ling).reshape(1, -1))
            X_test_now_aud = X_test.iloc[index]
            y_pred_aud = rfaud.predict_proba(np.array(X_test_now_aud).reshape(1, -1))
            #print(y_pred_aud)
            total = [(x + y)/2 for x, y in zip(y_pred_ling[0], y_pred_aud[0])]
            actual_pred = int(np.where(total == max(total))[0][0])
            #print(total)
            #print(actual_pred)
            if actual_pred == 0:
                total_preds.append('negative')
            else:
                total_preds.append('positive')
            
        print((index/len(y_pred_all))* 100)
        index += 1
    return total_preds

    
def all_around_fusion_cv_all_s_2(k,
                        x_ff, y_ff,
                        xl, yl,
                        x,y):    
    '''Performs k-fold CV of All-Around Fusion model,
    Returns: Average classification metrics
    including: weighted precision and F1-Score,
    accuracy and macro recall'''
    rfaud = RandomForestClassifier(n_estimators=600, min_samples_split=4,
                                 min_impurity_decrease=0.0, max_features = 'sqrt',max_depth = None, 
                                 criterion = 'entropy', bootstrap = False)
    rfling = RandomForestClassifier(n_estimators=600, min_samples_split=4,
                                 min_impurity_decrease=0.0, max_features = 'sqrt',max_depth = None, 
                                 criterion = 'entropy', bootstrap = False)
    rfall = RandomForestClassifier(n_estimators=600, min_samples_split=4,
                                 min_impurity_decrease=0.0, max_features = 'sqrt',max_depth = None, 
                                 criterion = 'entropy', bootstrap = False)
    result = []
    result_recall = []
    result_roc = []
    result_precision = []
    result_f1 = []
    kf = KFold(n_splits=k, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(x, y):
        print('Fold no : ', fold_count)
        X_trainff, X_testff = x_ff.iloc[train_index], x_ff.iloc[test_index]
        y_trainff, y_testff = y_ff[train_index], y_ff[test_index]
        
        X_trainl, X_testl = xl.iloc[train_index], xl.iloc[test_index]
        y_trainl, y_testl = yl[train_index], yl[test_index]
        
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        rfling.fit(X_trainl,y_trainl)
        rfall.fit(X_trainff, y_trainff)
        rfaud.fit(X_train, y_train)

        preds_fu = all_around_fusion_all_s_2(X_trainff, X_testff,
                        y_trainff, y_testff,
                        X_trainl, X_testl,
                        y_trainl, y_testl,
                        X_train, X_test,
                        y_train, y_test,
                        rfaud, rfling, rfall)

        print(classification_report(y_test,preds_fu))
        print(accuracy_score(y_test, preds_fu))
        print(f1_score(y_test, preds_fu, average = 'weighted'))
        result.append(accuracy_score(y_test, preds_fu))
        result_recall.append(recall_score(y_test, preds_fu, average="macro"))
        result_precision.append(precision_score(y_test, preds_fu, average="weighted"))
        result_f1.append(f1_score(y_test, preds_fu, average = 'weighted'))
        
    return sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)

all_around_fusion_cv_all_s_2(3,
                        x_ff, y_ff,
                        xl, yl,
                        x,y)
'''
Returns:
(0.7482042321879246,
 0.7004486222094757,
 0.7438358693905753,
 0.7392874773041423)
'''

# DECISION FUSION JIA


def decision_fusion(X_trainff, X_testff,
                        y_trainff, y_testff,
                        X_trainl, X_testl,
                        y_trainl, y_testl,
                        X_train, X_test,
                        y_train, y_test,
                        rfaud, rfling):    
    '''Performs Jia Decision-Level Fusion.
    Returns: classifications for test data-set.'''
    total_preds = []
    y_pred_all = rfaud.predict_proba(X_test)
    index = 0
    for preds in list(y_pred_all):
        #print(preds)
        #print(preds)
        actual_pred = int(np.where(preds == max(preds))[0][0])
        if actual_pred == 0:
            total_preds.append('negative')
        else:
            X_test_now_ling = X_testl.iloc[index]
            y_pred_ling = rfling.predict(np.array(X_test_now_ling).reshape(1, -1))
            if y_pred_ling == 'negative':
                total_preds.append('negative')
            else:
                total_preds.append('positive')
        print((index/len(y_pred_all))* 100)
        index += 1
    return total_preds

def all_around_fusion_cv_all_jia(k,
                        x_ff, y_ff,
                        xl, yl,
                        x,y):    
    '''Performs k-fold CV of Jia Decision-Level Fusion model,
    Returns: Average classification metrics
    including: weighted precision and F1-Score,
    accuracy and macro recall'''
    rfaud = RandomForestClassifier(n_estimators=600, min_samples_split=4,
                                 min_impurity_decrease=0.0, max_features = 'sqrt',max_depth = None, 
                                 criterion = 'entropy', bootstrap = False)
    rfling = RandomForestClassifier(n_estimators=600, min_samples_split=4,
                                 min_impurity_decrease=0.0, max_features = 'sqrt',max_depth = None, 
                                 criterion = 'entropy', bootstrap = False)
    result = []
    result_recall = []
    result_precision = []
    result_f1 = []
    kf = KFold(n_splits=k, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(x, y):
        print('Fold no : ', fold_count)
        X_trainff, X_testff = x_ff.iloc[train_index], x_ff.iloc[test_index]
        y_trainff, y_testff = y_ff[train_index], y_ff[test_index]
        X_trainl, X_testl = xl.iloc[train_index], xl.iloc[test_index]
        y_trainl, y_testl = yl[train_index], yl[test_index]
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        rfaud.fit(X_train, y_train)
        rfling.fit(X_trainl, y_trainl)
        preds_fu = decision_fusion(X_trainff, X_testff,
                        y_trainff, y_testff,
                        X_trainl, X_testl,
                        y_trainl, y_testl,
                        X_train, X_test,
                        y_train, y_test,
                        rfaud, rfling)
        print(classification_report(y_test,preds_fu))
        print(accuracy_score(y_test, preds_fu))
        print(f1_score(y_test, preds_fu, average = 'weighted'))
        print(confusion_matrix(y_test,preds_fu))
        result.append(accuracy_score(y_test, preds_fu))
        result_recall.append(recall_score(y_test, preds_fu, average="macro"))
        result_precision.append(precision_score(y_test, preds_fu, average="weighted"))
        result_f1.append(f1_score(y_test, preds_fu, average = 'weighted'))
    return sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)


all_around_fusion_cv_all_jia(3,
                        x_ff, y_ff,
                        xl, yl,
                        x,y)
'''
Returns:
(0.6734614637934381,
0.5490356321423243,
0.7142471090025956,
0.5822145016885819)
'''

 # DECISION FUSION ABBURI
 def all_around_fusion_all_abburi(X_trainff, X_testff,
                        y_trainff, y_testff,
                        X_trainl, X_testl,
                        y_trainl, y_testl,
                        X_train, X_test,
                        y_train, y_test,
                        rfaud, rfling):  
    '''Performs Abburi Decision-Level Fusion.
    Returns: classifications for test data-set.'''   
    total_preds = []
    y_pred_all = rfling.predict_proba(X_testl)
    index = 0
    for preds in list(y_pred_all):

        X_test_now_aud = X_test.iloc[index]
        y_pred_aud = rfaud.predict_proba(np.array(X_test_now_aud).reshape(1, -1))
        #print(y_pred_aud)
        total = [(x + y)/2 for x, y in zip(y_pred_aud[0], preds)]
        actual_pred = int(np.where(total == max(total))[0][0])
        #print(total)
        #print(actual_pred)
        if actual_pred == 0:
            total_preds.append('negative')
        else:
            total_preds.append('positive')
            
        print((index/len(y_pred_all))* 100)
        index += 1
    return total_preds


def all_around_fusion_cv_abburi(k,
                        x_ff, y_ff,
                        xl, yl,
                        x,y):    
    '''Performs k-fold CV of Abburi Decision-Level Fusion.
    Returns: Average classification metrics
    including: weighted precision and F1-Score,
    accuracy and macro recall'''
    rfaud = RandomForestClassifier(n_estimators=600, min_samples_split=4,
                                 min_impurity_decrease=0.0, max_features = 'sqrt',max_depth = None, 
                                 criterion = 'entropy', bootstrap = False)
    rfling = RandomForestClassifier(n_estimators=600, min_samples_split=4,
                                 min_impurity_decrease=0.0, max_features = 'sqrt',max_depth = None, 
                                 criterion = 'entropy', bootstrap = False)
    result = []
    result_recall = []
    result_precision = []
    result_f1 = []
    kf = KFold(n_splits=k, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(x, y):
        print('Fold no : ', fold_count)
        X_trainff, X_testff = x_ff.iloc[train_index], x_ff.iloc[test_index]
        y_trainff, y_testff = y_ff[train_index], y_ff[test_index]
        
        X_trainl, X_testl = xl.iloc[train_index], xl.iloc[test_index]
        y_trainl, y_testl = yl[train_index], yl[test_index]
        
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        rfling.fit(X_trainl,y_trainl)
        rfaud.fit(X_train, y_train)

        preds_fu = all_around_fusion_all_abburi(X_trainff, X_testff,
                        y_trainff, y_testff,
                        X_trainl, X_testl,
                        y_trainl, y_testl,
                        X_train, X_test,
                        y_train, y_test,
                        rfaud, rfling)

        print(classification_report(y_test,preds_fu))
        print(accuracy_score(y_test, preds_fu))
        print(f1_score(y_test, preds_fu, average = 'weighted'))
        result.append(accuracy_score(y_test, preds_fu))
        result_recall.append(recall_score(y_test, preds_fu, average="macro"))
        result_precision.append(precision_score(y_test, preds_fu, average="weighted"))
        result_f1.append(f1_score(y_test, preds_fu, average = 'weighted'))
        
    return sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)

'''
Returns:
(0.7342263638128519, 0.656239472326396, 0.739603555568602, 0.706404804218625)
'''
# DYNAMIC ALL AROUND FUSION
def all_around_dynamic(X_trainff, X_testff,
                        y_trainff, y_testff,
                        X_trainl, X_testl,
                        y_trainl, y_testl,
                        X_train, X_test,
                        y_train, y_test,
                        rfaud, rfling, rfall): 
    '''Performs Dynamic All-Around Fusion.
    Returns: classifications for test data-set.'''  
    indexer = 0
    predictions_all = []
    kf = KFold(n_splits=3, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(X_trainff, y_trainff):
        print('Fold no : ', indexer)
        X_train_to_remove, X_test_to_remove = X_trainff.iloc[train_index], X_trainff.iloc[test_index]

        y_train_to_remove, y_test_to_remove = y_trainff.iloc[train_index], y_trainff.iloc[test_index]

        rfall.fit(X_train_to_remove, y_train_to_remove)
        predictions = rfall.predict(X_test_to_remove)
        predictions_all.append(predictions)
        indexer += 1
    predictions_ff = np.concatenate(predictions_all).ravel().tolist()
    X_trainl_st2 = X_trainl.iloc[[pr != 'positive' for pr in predictions_ff]]
    y_trainl_st2 = y_trainl.iloc[[pr != 'positive' for pr in predictions_ff]]
    X_train_st2 = X_train.iloc[[pr != 'positive' for pr in predictions_ff]]
    y_train_st2 = y_train.iloc[[pr != 'positive' for pr in predictions_ff]]
    
    total_preds = all_around_fusion_all_s_2(X_trainff, X_testff,
                        y_trainff, y_testff,
                        X_trainl_st2, X_testl,
                        y_trainl_st2, y_testl,
                        X_train_st2, X_test,
                        y_train_st2, y_test,
                        rfaud, rfling, rfall)
    return total_preds

def all_around_fusion_cv_all_dynamic_aa(k,
                        x_ff, y_ff,
                        xl, yl,
                        x,y):    
    '''Performs k-fold CV of Dynamic All-Around Fusion.
    Returns: Average classification metrics
    including: weighted precision and F1-Score,
    accuracy and macro recall'''
    rfaud = RandomForestClassifier(n_estimators=600, min_samples_split=4,
                                 min_impurity_decrease=0.0, max_features = 'sqrt',max_depth = None, 
                                 criterion = 'entropy', bootstrap = False)
    rfling = RandomForestClassifier(n_estimators=600, min_samples_split=4,
                                 min_impurity_decrease=0.0, max_features = 'sqrt',max_depth = None, 
                                 criterion = 'entropy', bootstrap = False)
    rfall = RandomForestClassifier(n_estimators=600, min_samples_split=4,
                                 min_impurity_decrease=0.0, max_features = 'sqrt',max_depth = None, 
                                 criterion = 'entropy', bootstrap = False)
    result = []
    result_roc = []
    result_recall = []
    result_precision = []
    result_f1 = []
    kf = KFold(n_splits=k, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(x, y):
        print('Fold no : ', fold_count)
        X_trainff, X_testff = x_ff.iloc[train_index], x_ff.iloc[test_index]
        y_trainff, y_testff = y_ff[train_index], y_ff[test_index]
        
        X_trainl, X_testl = xl.iloc[train_index], xl.iloc[test_index]
        y_trainl, y_testl = yl[train_index], yl[test_index]
        
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        rfling.fit(X_trainl,y_trainl)
        rfall.fit(X_trainff, y_trainff)
        rfaud.fit(X_train, y_train)

        preds_fu = all_around_dynamic(X_trainff, X_testff,
                        y_trainff, y_testff,
                        X_trainl, X_testl,
                        y_trainl, y_testl,
                        X_train, X_test,
                        y_train, y_test,
                        rfaud, rfling, rfall)

        print(classification_report(y_test,preds_fu))
        print(accuracy_score(y_test, preds_fu))
        print(f1_score(y_test, preds_fu, average = 'weighted'))
        result.append(accuracy_score(y_test, preds_fu))
        result_recall.append(recall_score(y_test, preds_fu, average="macro"))
        result_precision.append(precision_score(y_test, preds_fu, average="weighted"))
        result_f1.append(f1_score(y_test, preds_fu, average = 'weighted'))
    return  sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)

all_around_fusion_cv_all_dynamic_aa(3,
                        x_ff, y_ff,
                        xl, yl,
                        x,y)
'''
Returns:
(0.7313143079013784, 0.681834330461926, 0.7246650711003163, 0.7214792118571927)
'''

#Perform McNemars test with continuity correction on predictions of ling Tfidf and All-Around model
rfaud = RandomForestClassifier(n_estimators=600, min_samples_split=4,
                                 min_impurity_decrease=0.0, max_features = 'sqrt',max_depth = None, 
                                 criterion = 'entropy', bootstrap = False)
rfling = RandomForestClassifier(n_estimators=600, min_samples_split=4,
                             min_impurity_decrease=0.0, max_features = 'sqrt',max_depth = None, 
                             criterion = 'entropy', bootstrap = False)
rfall = RandomForestClassifier(n_estimators=600, min_samples_split=4,
                             min_impurity_decrease=0.0, max_features = 'sqrt',max_depth = None, 
                             criterion = 'entropy', bootstrap = False)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
X_trainl, X_testl, y_trainl, y_testl = train_test_split(xl, yl, test_size=0.30)
X_trainff, X_testff, y_trainff, y_testff = train_test_split(x_ff, y_ff, test_size=0.30)
rfling.fit(X_trainl,y_trainl)
rfall.fit(X_trainff, y_trainff)
rfaud.fit(X_train, y_train)
preds_fu = all_around_fusion_all_s_2(X_trainff, X_testff,
                y_trainff, y_testff,
                X_trainl, X_testl,
                y_trainl, y_testl,
                X_train, X_test,
                y_train, y_test,
                rfaud, rfling, rfall)
preds_ling = rfling.predict(X_testl)

from statsmodels.stats.contingency_tables import mcnemar

print(mcnemar(confusion_matrix(preds_fu,preds_ling), exact = False, correction = True))

'''
Returns:
pvalue      1.954938609508094e-21
statistic   90.39058823529412
'''