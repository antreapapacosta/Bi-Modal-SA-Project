import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import librosa
import librosa.display
import librosa.display
import os
import wavio
import random
import scipy.io.wavfile
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
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
from math import log
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

transcription_dir = 'C:\\Users\\Andrea\\Desktop\\IEMOCAP_data\\IEMOCAP_full_release\\Session1\\dialog\\transcriptions'


import os
all_files = os.listdir(transcription_dir)
list_sessions = [1,2,3,4,5]
def get_transcriptions(session):
    ''' 
    Function to get transcriptions of IEMOCAP data-set.
    Locates all utterances and their associated Identifier (file name) in directory
    Returns: 
    Data-frame of utterances and identifiers
    '''
    transcription_dir = 'C:\\Users\\Andrea\\Desktop\\IEMOCAP_data\\IEMOCAP_full_release\\Session' + str(session) + '\\dialog\\transcriptions'
    all_files = os.listdir(transcription_dir)
    dn = []
    for file in all_files:
        #print(file)
        file_path = transcription_dir + '//' + file
        df = pd.read_csv(file_path, sep=": ", header=None, engine='python')
        #print(df.columns)
        df.columns = ['Identifier', 'Utterance']
        df['Identifier'] = df['Identifier'].str.split(' ').str[0]
        #print(df)
        dn.append(df)
    dn = pd.concat(dn, axis=0)
    return dn


final = []
for session in list_sessions:
    df = get_transcriptions(session)
    final.append(df)
final = pd.concat(final, axis=0)
final.shape


final_df = final[final["Identifier"].str.contains('Ses')].reset_index().iloc[:,1:]


final_df['wav_path'] = 0
final_df['emo_path'] = 0

# gather emotion label and wav paths of all utterances
glob = 'C:\\Users\\Andrea\\Desktop\\IEMOCAP_data\\IEMOCAP_full_release\\Session' 
for i in range(len(final_df)):
    local_identifier = final_df["Identifier"][i]
    global_identifier = final_df["Identifier"][i][:-5]
    session_no = global_identifier[4]
    file_wav_dir = glob + session_no + '\\sentences\\wav\\' + global_identifier + '\\' + local_identifier + '.wav'
    file_emo_dir = glob + session_no + '\\dialog\\EmoEvaluation\\' + global_identifier +  '.txt'
    final_df['wav_path'][i] = file_wav_dir
    final_df['emo_path'][i] = file_emo_dir

# gather emotion labels of all utterances
dialogues = set(list(final_df['emo_path']))
all_emotions = []
lengths = []
counter = 1
identifiers_list = []
for dialogue in dialogues:
    counter +=1 
    print(counter)
    with open(dialogue) as f:
        lines = f.readlines()
    all_identifiers = list(final_df['Identifier'])
    emotions = []
    for line in lines:
        for identifier in all_identifiers:
            if identifier in line:
                print(identifier)
                print(line)
                print(line.split('\t')[2])
                identifiers_list.append(identifier)
                emotions.append(line.split('\t')[2])
                lengths.append(line.split('\t')[1])
    all_emotions.append(emotions)
   

all_identifiers_tokeep = list(set(final_df['Identifier']).intersection(set(lengths)))
df_final = final_df[final_df.isin(all_identifiers_tokeep).any(1)]
d = {'Emotions':sum(all_emotions, []),'Identifier':lengths}
merged = df_final.merge(pd.DataFrame(d, columns=['Emotions','Identifier']),
                        on = 'Identifier', how = 'inner', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

merged_final = merged.drop('emo_path', axis = 1)
emotions_unique = list(set(merged_final['Emotions']))
emotions_unique.remove('xxx')
emotions_unique.remove('oth')

# emotion to sentiment conversion according to Robinson scale
my_dict = {'fru': 'negative', 'neu': 'neutral', 'sad': 'negative',
          'ang': 'negative', 'sur': 'positive', 'fea' : 'negative', 
          'hap': 'positive', 'dis': 'negative', 'exc': 'positive'}
final_merged_df = merged_final[merged_final.isin(emotions_unique).any(1)].reset_index().iloc[:,1:]
sentiment_m_df = final_merged_df.replace(my_dict)
linguistic = sentiment_m_df
data_meld =linguistic

# gather word count
word_count = []
for i in range(len(linguistic)):
    word_count.append(len(linguistic['Utterance'][i]))


linguistic['WordCount'] = word_count

linguistic = linguistic.drop('wav_path', axis = 1)
linguistic.columns = ['Identifier', 'Utterance', 'Sentiment', 'WordCount']
sentiment_m_df['WordCount'] = word_count

############ read data meld
# read utterances of MELD data (see Meld.py)
data_meld_dir = "C:\\Users\\Andrea\\Desktop\\Project\\\IEMOCAP\\final_datameld.csv"
data_meld = pd.read_csv(data_meld_dir).iloc[:,1:].drop(['Speaker', 'Duration'], axis =1 )
data_meld_new = data_meld.drop_duplicates().reset_index().iloc[:,1:]
joined_meld_i = pd.concat([linguistic, data_meld_new]).reset_index().iloc[:,1:]
joined_meld_i = joined_meld_i.drop_duplicates().reset_index().iloc[:,1:]
pos_neg_all = joined_meld_i[joined_meld_i['Sentiment'].isin(['positive','negative'])].reset_index().iloc[:,1:]


############ convert Combined utterances to linguistic features
from sklearn.feature_extraction.text import TfidfVectorizer
import inflect
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import string
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
wml = WordNetLemmatizer()
p = inflect.engine()

def convert_word(sentence):
    ''' Function to pre-process words by:
    Removing punctuation,
    Convering words to lowercase,
    Removing Stopwords,
    Converting numerical to word format for numbers,
    Lemmatizing words,
    Returns: filtered word'''    
    all_words = []
    filtered_words = []
    Stopwords = list(set(stopwords.words('english')))
    sentence = re.sub(r'[^\w\s]','',sentence)
    tot_words = sentence.split(" ")
    # converts to lowercase
    all_words = [item.lower() for item in tot_words]
    # removes spaces
    all_words = list(filter(None, all_words))
    for word in all_words:
        if word in Stopwords:
            pass
            #print('Stop word is ' + word)
        else:
            if word.isdigit():
                count = all_words.index(word)
                all_words[count] = (p.number_to_words(word))
                filtered_words.append(p.number_to_words(word))
            else:
                count = all_words.index(word)
                all_words[count] = wml.lemmatize(word)
                filtered_words.append(wml.lemmatize(word))
    return  filtered_words


all_words = []
for line in range(len(joined_meld_i)):
    all_words.append(convert_word(joined_meld_i['Utterance'][line]))

ind = 0
tot = []
for x in all_words:
    if len(x)<=3:
        #print(ind)
        tot.append(ind)
    ind+=1
    

desired = list(set(range(len(joined_meld_i))).difference(set(tot)))
all_words = [all_words[i] for i in desired]
unique_words = list(set(sum(all_words, [])))


def bag_o_words(all_words, unique_words):
    '''Function to convert list of all words and unique words
    to bag of word format. 
    Returns: pandas data-frame as BoW'''
    all_counts = []
    index = 0
    for sentence_words in all_words:
        total_count = []
        for word in unique_words:
            occurences = sentence_words.count(word)
            count = occurences
            total_count.append(count)
        all_counts.append(total_count)
        index += 1
    bow_corpus = pd.DataFrame(all_counts)
    bow_corpus.columns= unique_words
    return bow_corpus

bow_corpus = bag_o_words(all_words, unique_words)
WordCounts = list(joined_meld_i['WordCount'])
Sentiments = list(joined_meld_i['Sentiment'])
Identifiers = list(joined_meld_i['Identifier'])
Sentiments_list = [Sentiments[i] for i in desired]
Word_count_list = [WordCounts[i] for i in desired]
Identifiers_list = [Identifiers[i] for i in desired]

bow_corpus['Sentiment'] = Sentiments_list
bow_corpus['WordCount'] = Word_count_list
bow_corpus['Identifiers'] = Identifiers_list

def get_tfidf(all_words):
    '''
    Function to convert list of all words and unique words
    to TF-IDF format. 
    Returns: pandas data-frame as TF-IDF
    '''
    all_counts = []
    N = len(all_words)
    for sentence_words in all_words:
        n = (len(sentence_words))
        total_count = []
        counter = 0
        for word in unique_words:
            tf = sentence_words.count(word)/n
            idf = log(N/dfi[counter])
            count = tf * idf
            total_count.append(count)
            counter += 1
        all_counts.append(total_count)
    tfidf = pd.DataFrame(all_counts)
    tfidf.columns= unique_words
    return tfidf

dfi = []
for word in unique_words:
    dfi.append(len(bow_corpus[bow_corpus[word]>=1]))

tf_idf_corpus = get_tfidf(all_words)
WordCounts = list(joined_meld_i['WordCount'])
Sentiments = list(joined_meld_i['Sentiment'])
Identifiers = list(joined_meld_i['Identifier'])
tf_idf_corpus['Sentiment'] = Sentiments_list
tf_idf_corpus['WordCount'] = Word_count_list
tf_idf_corpus['Identifiers'] = Identifiers_list
tf_idf_corpus_final = tf_idf_corpus.drop_duplicates().reset_index().iloc[:,1:]
tf_idf_corpus_final.to_csv('C:\\Users\\Andrea\\Desktop\\Project\\IEMOCAP\\tfidf_ch2.csv')
bow_corpus.to_csv('C:\\Users\\Andrea\\Desktop\\Project\\IEMOCAP\\' + 'bow.csv')



###############################IEMOCAP Audio Features
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
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, AveragePooling1D, BatchNormalization, Input, Flatten, Dropout, Activation
from keras.utils import to_categorical, np_utils

sentiment_m_df.columns = ['Identifier', 'Utterance', 'wav_path', 'Sentiment', 'WordCount']

def play_sample(sentiment_sample, data_frame, play = False):
    '''
    Function to play audio sample of a specific emotion,
    to develop spectrograms and wave-plots.
    Audio is played only when play = True (default = False).
    Returns: 2 matplotlib.pyplot figures and audio sample 
    '''
    sentiment = data_frame['Sentiment']
    unique = list(set(sentiment))
    if sentiment_sample not in unique:
        return 'Sentiment not recognised, try another emotion.'
    else:
        all_em = list(data_frame[data_frame['Sentiment'] == sentiment_sample]['wav_path'])
        random_emotion = random.choices(all_em, k=1)
        file_path = random_emotion[0]
        if play == True:
            playsound(file_path)
        data, sampling_rate = librosa.load(file_path)
        plot = librosa.display.waveplot(data, sr=sampling_rate)
        plt.title('Sentiment : ' + sentiment_sample)
        plt.ylabel('Amplitude')
        f_trans = librosa.stft(data)
        Xdb = librosa.amplitude_to_db(abs(f_trans))
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='hz') 
        #If to pring log of frequencies  
        #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar()
        plt.title('Sentiment : ' + sentiment_sample)
        plt.show()
        return plot

play_sample('positive', sentiment_m_df, play=False)



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
    for col in range(1,num_coeff+1):
        name = 'coeff' + str(col)
        df[name] = 0
    return df


df = create_df(30, sentiment_m_df)

df = df.drop(['Utterance', 'Identifier'], axis = 1)

from pytictoc import TicToc
t = TicToc() #create instance of class


def get_features(df, num_coeff):
    '''
    Function to perform feature construction for acoustic features.
    Returns:
    Dataframe with acoustic features.
    '''
    k = 0
    allpaths = df['wav_path']
    t.tic() #Start timer
    while k <= (df.shape[0]-1):
        for fp in allpaths:
            tol = 1e-3
            data, sr = librosa.load(fp)
            df.iloc[k,3] = percentage_silence(data, sr, tol)
            df.iloc[k,4] = total_silence_seconds(data, sr, tol)
            df.iloc[k,5] = max_silent_period(data, sr, tol)
            df.iloc[k,6] = sum(librosa.zero_crossings(data, pad=False))
            df.iloc[k,7] = librosa.beat.tempo(data)[0]
            spectral_centroids = librosa.feature.spectral_centroid(data, sr=sr)[0]
            df.iloc[k,8] = sum(spectral_centroids)/len(spectral_centroids)
            change_beats = []
            if len(spectral_centroids) != 0:
                for i in range(1,len(spectral_centroids)+1):
                    if i != len(spectral_centroids):
                        change_beats.append(spectral_centroids[i]-spectral_centroids[i-1])

            df.iloc[k,9] = sum(change_beats)/len(change_beats)
            df.iloc[k,10] = sum(np.squeeze(librosa.feature.rms(y=data)))/len(np.squeeze(librosa.feature.rms(y=data)))
            mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=num_coeff).T,axis=0) 
            #print(len(mfccs)
            for col in range(1,num_coeff + 1):
                column_index = 10 + col
                #print(column_index)
                counter = col-1
                
                #print(counter)
                #print(df.iloc[1, column_index] )
                df.iloc[k, column_index] = mfccs[counter]
            k += 1
            print((k/df.shape[0]) * 100)
            print(k)
            #print(df.head(k))
    t.toc() #Time elapsed since t.tic()
    return df

trail_1 = get_features(df, 30)
trail_1.to_csv('audio_iemocap.csv')

#########################MELD AUDIO FEATURES###############
directory = 'C:\\Users\\Andrea\\Desktop\\Project\\MELD\\'

data_meld_dir = "C:\\Users\\Andrea\\Desktop\\Project\\\IEMOCAP\\final_datameld.csv"
data_meld = pd.read_csv(data_meld_dir).iloc[:,1:].drop(['Utterance'], axis =1 )
data_meld_new = data_meld.drop_duplicates().reset_index().iloc[:,1:]

os.chdir('C:\\Users\\Andrea\\Desktop\\Project\\MELD')
mylist = []
Train_data_path = 'C:\\Users\\Andrea\\Desktop\\Project\\MELD\\wav'

for path, subdirs, files in os.walk(Train_data_path):
    for name in files:
        if 'wav' in name:
            mylist.append(os.path.join(path, name))

new_data_meld = data_meld_new.drop_duplicates().reset_index()
index_to_remove = []
for i in range(len(new_data_meld)):
    file_name  = directory + 'wav\\'+ new_data_meld.iloc[i]['Identifier']
    if file_name not in mylist:
        print(file_name)
        index_to_remove.append(i)

audio_df = new_data_meld.drop(index_to_remove, axis = 0).reset_index().iloc[:,2:]

def play_sample(sentiment_sample, data_frame, directory, play = False):
    '''
    Function to play audio sample of a specific emotion,
    to develop spectrograms and wave-plots.
    Audio is played only when play = True (default = False).
    Returns: 2 matplotlib.pyplot figures and audio sample 
    '''    
    sentiment = data_frame['Sentiment']
    unique = list(set(sentiment))
    if sentiment_sample not in unique:
        return 'Sentiment not recognised, try another emotion.'
    else:
        all_em = list(data_frame[data_frame['Sentiment'] == sentiment_sample]['Identifier'])
        random_emotion = random.choices(all_em, k=1)
        file_path = directory +'wav\\'+  random_emotion[0]
        print(file_path)
        if play == True:
            playsound(file_path)
        data, sampling_rate = librosa.load(file_path)
        plot = librosa.display.waveplot(data, sr=sampling_rate)
        plt.title('Sentiment : ' + sentiment_sample)
        plt.ylabel('Amplitude')
        f_trans = librosa.stft(data)
        Xdb = librosa.amplitude_to_db(abs(f_trans))
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='log') 
        #If to pring log of frequencies  
        #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Sentiment : ' + sentiment_sample)
        plt.show()
        return plot


play_sample('negative', audio_df, directory, play=True)

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
    for col in range(1,num_coeff+1):
        name = 'coeff' + str(col)
        df[name] = 0
    return df

from pytictoc import TicToc
t = TicToc() #create instance of class


def get_features(df, num_coeff):
    '''
    Function to perform feature construction for acoustic features.
    Returns:
    Dataframe with acoustic features.
    '''
    k = 0
    allpaths = directory + 'wav\\' + df['Identifier']
    print(allpaths)
    t.tic() #Start timer
    while k <= (df.shape[0]-1):
        for fp in allpaths:
            tol = 1e-3
            data, sr = librosa.load(fp)
            print(fp)
            print(k)
            df.iloc[k,5] = percentage_silence(data, sr, tol)
            df.iloc[k,6] = total_silence_seconds(data, sr, tol)
            df.iloc[k,7] = max_silent_period(data, sr, tol)
            df.iloc[k,8] = sum(librosa.zero_crossings(data, pad=False))
            df.iloc[k,9] = librosa.beat.tempo(data)[0]
            spectral_centroids = librosa.feature.spectral_centroid(data, sr=sr)[0]
            df.iloc[k,10] = sum(spectral_centroids)/len(spectral_centroids)
            change_beats = []
            if len(spectral_centroids) != 0:
                for i in range(1,len(spectral_centroids)+1):
                    if i != len(spectral_centroids):
                        change_beats.append(spectral_centroids[i]-spectral_centroids[i-1])

            df.iloc[k,11] = sum(change_beats)/len(change_beats)
            df.iloc[k,12] = sum(np.squeeze(librosa.feature.rms(y=data)))/len(np.squeeze(librosa.feature.rms(y=data)))
            mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=num_coeff).T,axis=0) 
            #print(len(mfccs)
            for col in range(1,num_coeff + 1):
                column_index = 12 + col
                #print(column_index)
                counter = col-1
                
                #print(counter)
                #print(df.iloc[1, column_index] )
                df.iloc[k, column_index] = mfccs[counter]
            k += 1
            print(k)
            #print(df.head(1))
    t.toc() #Time elapsed since t.tic()
    return df

df = create_df(30, audio_df)
trail_1 = get_features(df, 30)
trail_1.to_csv('C:\\Users\\Andrea\\Desktop\\Project\\MELD\\audio_meld_c.csv')





