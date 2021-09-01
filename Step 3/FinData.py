# FinData Analysis
import re
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import inflect
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB 
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from math import log
from nltk.stem.porter import PorterStemmer
import re
from sklearn.model_selection import RandomizedSearchCV
import string
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
wml = WordNetLemmatizer()
p = inflect.engine()

## The data cleaning and data reading process is ommited for data confidentiality reasons

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

## pre-process FinData 
all_words = []
for line in range(len(complete_df)):
    all_words.append(convert_word(complete_df['Utterance'][line]))

# only keep utterances of less than 3 words
ind = 0
tot = []
for x in all_words:
    if len(x)<=3:
        tot.append(ind)
    ind+=1

# find unique words
desired = list(set(range(len(complete_df))).difference(set(tot)))

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

dfi = []
for word in unique_words:
    dfi.append(len(bow_corpus[bow_corpus[word]>=1]))

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

tf_idf_corpus = get_tfidf(all_words)

# convert numerical labels to categorical
ratings = list(complete_df['staff_rating'])
categorical_rating = []
for rating in ratings:
    if int(rating) <= 8:
        categorical_rating.append('negative')
    else:
        categorical_rating.append('positive')

complete_df['Sentiment'] = categorical_rating

tf_idf['Sentiment'] = categorical_rating

# extract audio features from data-set
audio_df = complete_df.iloc[:,2:].drop('staff_rating', axis = 1)
# define linguistic and ff features
ling_df = tf_idf.iloc[:,1:]
ff_df = pd.concat([audio_df.drop('Sentiment',axis =1 ), ling_df], axis = 1)

# extract first principal component
pca = PCA(n_components=1)
pr_component = pca.fit_transform(audio_df.drop('Sentiment',axis=1))
pca_df = pd.DataFrame(data = pr_component)
pca_df_final = pca_df.add_prefix('PrincipalComponent')
pca_df_final
# obtain the total variance explained by pca component
print(round(sum(list(pca.explained_variance_ratio_))*100, 2))

# define ff features with pca and linguistic data-set
ff_df_pca = pd.concat([pca_df_final, ling_df], axis = 1)
df_ling = tf_idf.iloc[:,1:]

## RF training for linguistic model
x_ling, y_ling = df_ling.drop('Sentiment', axis = 1), df_ling['Sentiment']

# define test and train data-sets
X_train, X_test, y_train, y_test = train_test_split(tf_idf.drop('Sentiment', axis = 1), tf_idf['Sentiment'],
                                                    test_size = 0.30)
sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
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
                               n_iter = 200,
                               cv = 3,
                               verbose = 10, 
                               n_jobs = -1,
                               scoring = "f1_weighted")

classifier_rf_random.fit(X_train_res, y_train_res)
# view optimal hyper-parameters
classifier_rf_random.best_params_
'''#RETURNS:
{'n_estimators': 1500, 'min_samples_split': 7, 'min_impurity_decrease': 0.0, 
'max_features': 'auto', 'max_depth': 45, 'criterion': 'gini', 'bootstrap': False}'''
## get RF CV metrics
def cross_validation_rf(k, x, y):
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
    kf = StratifiedKFold(n_splits=k, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Perform SMOTE algorithm for better data balance
        sm = SMOTE(random_state=2)
        X_train_res, y_train_res = sm.fit_resample(x_train, y_train.ravel())
        # define optimal RF models
        rf_random_all= RandomForestClassifier(n_estimators = 1500,
                    min_samples_split= 7, 
				    min_impurity_decrease=0.0, max_features='auto',
			        max_depth= 45, criterion= 'gini', bootstrap= False)
        rf_random_all.fit(X_train_res, y_train_res.ravel())
        y_pred = rf_random_all.predict(x_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test,y_pred))
        print(accuracy_score(y_test, y_pred))
        result.append(accuracy_score(y_test, y_pred))
        result_recall.append(recall_score(y_test, y_pred, average="macro"))
        result_precision.append(precision_score(y_test, y_pred, average="weighted"))
        result_f1.append(f1_score(y_test, y_pred, average = 'weighted'))
    return sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)


cross_validation_rf(3, x_ling, y_ling)
''' 
Returns:
(0.7211172892310617, 0.5035460992907801, 0.6119852284869545, 0.6062370448065468)
'''


## Training of NB Multi-nomial

def cross_validation_nb(k, x, y):
    '''Performs k-fold CV of Multi-nomial NB model 
    on linguistic features
    Returns: Average classification metrics
    including: weighted precision and F1-Score,
    accuracy and macro recall'''
    result = []
    result_recall = []
    result_precision = []
    result_f1 = []
    actual_result = []
    kf = StratifiedKFold(n_splits=k, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        nb_model = MultinomialNB(alpha=1, fit_prior=True)
        y_train, y_test = y[train_index], y[test_index]
        # Perform SMOTE algorithm for better data balance
        sm = SMOTE(random_state=2)
        X_train_res, y_train_res = sm.fit_resample(x_train, y_train.ravel())
        nb_model.fit(X_train_res, y_train_res)
        y_pred = nb_model.predict(x_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test,y_pred))
        print(accuracy_score(y_test, y_pred))
        result.append(accuracy_score(y_test, y_pred))
        result_recall.append(recall_score(y_test, y_pred, average="macro"))
        result_precision.append(precision_score(y_test, y_pred, average="weighted"))
        result_f1.append(f1_score(y_test, y_pred, average="weighted"))
    return sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)

cross_validation_nb(3, x_ling, y_ling)
'''
Returns:
 (0.5060474289516206, 0.5377124728913897, 0.6280623324358738, 0.5260181973690431)

'''

# KNN LING

#List Hyperparameters that we want to tune.
leaf_size = list(range(1,50))
n_neighbors = list(range(1, 30))
hyper_knn = dict(leaf_size=leaf_size, n_neighbors=n_neighbors)
knn_model = KNeighborsClassifier()
knn_model_random = RandomizedSearchCV(knn_model,
                                      hyper_knn,
                                      cv=3,
                                      verbose = 10,
                                      n_iter = 100,
                                      scoring = "f1_weighted")
knn_model_random.fit(X_train_res, y_train_res)
print('Best leaf_size:', knn_model_random.best_estimator_.get_params()['leaf_size'])
print('Best p:', knn_model_random.best_estimator_.get_params()['p'])
print('Best n_neighbors:', knn_model_random.best_estimator_.get_params()['n_neighbors'])

'''
Returns:
Best leaf_size: 4
Best p: 2
Best n_neighbors: 7
'''
def cross_validation_knn_ling(k, x, y, p, leaf_size, n_neighbours ):
    '''Performs k-fold CV of KNN model
    using p, leaf_size and n_neighbours hyper-parameters
    Returns: Average classification metrics
    including: weighted precision and F1-Score,
    accuracy and macro recall'''
    result = []
    result_recall = []
    result_precision = []
    result_f1 = []
    actual_result = []
    kf = StratifiedKFold(n_splits=k, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        knn_model = KNeighborsClassifier(leaf_size = leaf_size, p = p, n_neighbors = n_neighbours)
        y_train, y_test = y[train_index], y[test_index]
        # Perform SMOTE algorithm for better data balance
        sm = SMOTE(random_state=2)
        X_train_res, y_train_res = sm.fit_resample(x_train, y_train.ravel())
        scaler = preprocessing.StandardScaler().fit(X_train_res)
        X_scaled_train = scaler.transform(X_train_res)
        X_scaled_test = scaler.transform(x_test)
        knn_model.fit(X_scaled_train, y_train_res)
        y_pred = knn_model.predict(X_scaled_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test,y_pred))
        print(accuracy_score(y_test, y_pred))
        result.append(accuracy_score(y_test, y_pred))
        result_recall.append(recall_score(y_test, y_pred, average="macro"))
        result_precision.append(precision_score(y_test, y_pred, average="weighted"))
        result_f1.append(f1_score(y_test, y_pred, average="weighted"))
    return sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)


cross_validation_knn_ling(3, x_ling, y_ling, 2, 4, 7)
'''
Returns:
(0.28087871875297027, 0.5, 0.07889347828505548, 0.12318611243168127)
'''

##LR LING
parameters = {
    'C': np.linspace(1, 10, 10)
             }
lr = LogisticRegression()
lr_model = GridSearchCV(lr, parameters, cv=5, verbose=5,
                        n_jobs=3,scoring = "f1_weighted")
lr_model.fit(X_train_res, y_train_res.ravel())
lr_model.best_estimator_.get_params()['C']
'''Returns: 8.0'''

def cross_validation_lr(k, x, y):
    '''Performs k-fold CV of Linear Regression model,
    with optimal hyper-parameters using linguistic data-set.
    Returns: Average classification metrics
    including: weighted precision and F1-Score,
    accuracy and macro recall'''
    result = []
    result_recall = []
    result_precision = []
    result_f1 = []
    actual_result = []
    kf = StratifiedKFold(n_splits=k, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Perform SMOTE algorithm for better data balance
        sm = SMOTE(random_state=2)
        X_train_res, y_train_res = sm.fit_resample(x_train, y_train.ravel())
        lr = LogisticRegression(C=8,penalty='l2', verbose=0)
        lr.fit(X_train_res, y_train_res.ravel())
        y_pred = lr.predict(x_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test,y_pred))
        print(accuracy_score(y_test, y_pred))
        result.append(accuracy_score(y_test, y_pred))
        result_recall.append(recall_score(y_test, y_pred, average="macro"))
        result_precision.append(precision_score(y_test, y_pred, average="weighted"))
        result_f1.append(f1_score(y_test, y_pred, average = 'weighted'))
    return sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)

cross_validation_lr(3, x_ling, y_ling)
'''
Returns
(0.7390694800874442, 0.5613819529922045, 0.7113492313747884, 0.6745323409100602)
'''
#########################################Audio Models############
x_audio, y_audio= audio_df.drop('Sentiment', axis = 1), audio_df['Sentiment']

# RF Audio
X_train, X_test, y_train, y_test = train_test_split(audio_df.drop('Sentiment', axis = 1), audio_df['Sentiment'],
                                                    test_size = 0.30)
sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
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
                               n_iter = 200,
                               cv = 3,
                               verbose = 10, 
                               n_jobs = -1,
                               scoring = "f1_weighted")
classifier_rf_random.fit(X_train_res, y_train_res)
# view optimal hyper-parameters
classifier_rf_random.best_params_
'''
Returns: 
{'n_estimators': 600, 'min_samples_split': 2,
 'min_impurity_decrease': 0.0, 'max_features': 'log2',
 'max_depth': 45, 'criterion': 'gini', 'bootstrap': False}
'''
def cross_validation_rf_audio(k, x, y):
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
    kf = StratifiedKFold(n_splits=k, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Perform SMOTE algorithm for better data balance
        sm = SMOTE(random_state=2)
        X_train_res, y_train_res = sm.fit_resample(x_train, y_train.ravel())
        rf_random_all= RandomForestClassifier(n_estimators = 600,
                    min_samples_split= 2, 
				    min_impurity_decrease=0.0, max_features='log2',
			        max_depth= 45, criterion= 'gini', bootstrap= False)
        rf_random_all.fit(X_train_res, y_train_res.ravel())
        y_pred = rf_random_all.predict(x_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test,y_pred))
        print(accuracy_score(y_test, y_pred))
        result.append(accuracy_score(y_test, y_pred))
        result_recall.append(recall_score(y_test, y_pred, average="macro"))
        result_precision.append(precision_score(y_test, y_pred, average="weighted"))
        result_f1.append(f1_score(y_test, y_pred, average = 'weighted'))
    return sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)

cross_validation_rf_audio(3, x_audio, y_audio)
'''
Returns:
(0.5896896682824826, 0.5094359943731317, 0.603979459374845, 0.5944459065353124)
'''
# LR Audio model
parameters = {
    'C': np.linspace(1, 10, 10)
             }
lr = LogisticRegression()
lr_model = GridSearchCV(lr, parameters, cv=5, verbose=5,
                        n_jobs=3,scoring = "f1_weighted")
lr_model.fit(X_train_res, y_train_res.ravel())
lr_model.best_estimator_.get_params()['C']
'''Returns: 1.0'''

def cross_validation_lr_audio(k, x, y):
    '''Performs k-fold CV of Linear Regression model,
    with optimal hyper-parameters using audio features.
    Returns: Average classification metrics
    including: weighted precision and F1-Score,
    accuracy and macro recall'''
    result = []
    result_recall = []
    result_precision = []
    result_f1 = []
    actual_result = []
    kf = StratifiedKFold(n_splits=k, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Perform SMOTE algorithm for better data balance
        sm = SMOTE(random_state=2)
        X_train_res, y_train_res = sm.fit_resample(x_train, y_train.ravel())
        scaler = preprocessing.StandardScaler().fit(X_train_res)
        X_scaled_train = scaler.transform(X_train_res)
        X_scaled_test = scaler.transform(x_test)
        lr = LogisticRegression(C=1,penalty='l1', verbose=0, solver='liblinear')
        lr.fit(X_scaled_train, y_train_res.ravel())
        y_pred = lr.predict(X_scaled_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test,y_pred))
        print(accuracy_score(y_test, y_pred))
        result.append(accuracy_score(y_test, y_pred))
        result_recall.append(recall_score(y_test, y_pred, average="macro"))
        result_precision.append(precision_score(y_test, y_pred, average="weighted"))
        result_f1.append(f1_score(y_test, y_pred, average = 'weighted'))
    return sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)

cross_validation_lr_audio(3, x_audio, y_audio)

'''
Returns:
 (0.5877055412983557, 0.5793039192700701, 0.6583120446228623, 0.6073734848894521)
'''

# NB Audio Classifier

nb_model = GridSearchCV(estimator=GaussianNB(), param_grid=hyper_nb,
                            verbose=1, cv=10, n_jobs=-1, scoring = "accuracy")
nb_model.fit(X_train_res, y_train_res)
print(nb_model.best_estimator_)
'''
Returns:
GaussianNB(var_smoothing = 0.02848035868435802)
'''

def cross_validation_nb_audio(k, x, y):
    '''Performs k-fold CV of Gaussian Naive Bayes model,
    with optimal hyper-parameters using audio features.
    Returns: Average classification metrics
    including: weighted precision and F1-Score,
    accuracy and macro recall'''
    result = []
    result_recall = []
    result_precision = []
    result_f1 = []
    actual_result = []
    kf = StratifiedKFold(n_splits=k, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Perform SMOTE algorithm for better data balance
        sm = SMOTE(random_state=2)
        X_train_res, y_train_res = sm.fit_resample(x_train, y_train.ravel())
        rf_random_all= GaussianNB(var_smoothing = 0.02848035868435802)
        rf_random_all.fit(X_train_res, y_train_res.ravel())
        y_pred = rf_random_all.predict(x_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test,y_pred))
        print(accuracy_score(y_test, y_pred))
        result.append(accuracy_score(y_test, y_pred))
        result_recall.append(recall_score(y_test, y_pred, average="macro"))
        result_precision.append(precision_score(y_test, y_pred, average="weighted"))
        result_f1.append(f1_score(y_test, y_pred, average = 'weighted'))
    return sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)

cross_validation_nb_audio(3, x_audio, y_audio)
'''
Returns:
(0.5337182777302538, 0.5935769689154603, 0.6822893727190144, 0.5468030216497713)
'''

# Audio KNN
knn_model_random.fit(X_train_res, y_train_res)
print('Best leaf_size:', knn_model_random.best_estimator_.get_params()['leaf_size'])
print('Best p:', knn_model_random.best_estimator_.get_params()['p'])
print('Best n_neighbors:', knn_model_random.best_estimator_.get_params()['n_neighbors'])
'''
Returns:
Best leaf_size: 41
Best p: 2
Best n_neighbors: 1
'''
cross_validation_knn_ling(3, x_audio, y_audio, p = 2, leaf_size = 41, n_neighbours = 1 )
'''
Returns: 
# (0.5577654215378767, 0.4893709825528007, 0.5878151463801047, 0.5704979044278812)
'''

#####################################Fusion Models#########################
x_all, y_all = ff_df_pca.drop('Sentiment', axis = 1), ff_df_pca['Sentiment']

# Feature Fusion
def cross_validation_lr_ff(k, x, y):
    '''Performs k-fold CV of Feature Fusion model,
    using LR classifier.
    Returns: Average classification metrics
    including: weighted precision and F1-Score,
    accuracy and macro recall'''
    result = []
    result_recall = []
    result_precision = []
    result_f1 = []
    actual_result = []
    kf = StratifiedKFold(n_splits=k, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Perform SMOTE algorithm for better data balance
        sm = SMOTE(random_state=2)
        X_train_res, y_train_res = sm.fit_resample(x_train, y_train.ravel())
        scaler = preprocessing.StandardScaler().fit(X_train_res)
        X_scaled_train = scaler.transform(X_train_res)
        X_scaled_test = scaler.transform(x_test)
        lr = LogisticRegression(C=1,penalty='l1', verbose=0, solver='liblinear')
        lr.fit(X_scaled_train, y_train_res.ravel())
        y_pred = lr.predict(X_scaled_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test,y_pred))
        print(accuracy_score(y_test, y_pred))
        result.append(accuracy_score(y_test, y_pred))
        result_recall.append(recall_score(y_test, y_pred, average="macro"))
        result_precision.append(precision_score(y_test, y_pred, average="weighted"))
        result_f1.append(f1_score(y_test, y_pred, average = 'weighted'))
    return sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)
cross_validation_lr_ff(3, x_all, y_all)
'''
Returns:
(0.725002376199981, 0.5905698669480101, 0.6920502553403177, 0.69476163925692)

'''


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
    '''Performs k-fold CV of All-Around Fusion model.
    Returns: Average classification metrics
    including: weighted precision and F1-Score,
    accuracy and macro recall'''
    rfaud = LogisticRegression(C=1,penalty='l1', verbose=0, solver='liblinear')
    rfall= LogisticRegression(C=1,penalty='l1', verbose=0, solver='liblinear')
    rfling = LogisticRegression(C=1,penalty='l1', verbose=0, solver='liblinear')
    result = []
    result_recall = []
    result_precision = []
    result_f1 = []
    kf = StratifiedKFold(n_splits=k, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(x, y):
        print('Fold no : ', fold_count)
        X_trainff, X_testff = x_ff.iloc[train_index], x_ff.iloc[test_index]
        y_trainff, y_testff = y_ff[train_index], y_ff[test_index]
        X_trainl, X_testl = xl.iloc[train_index], xl.iloc[test_index]
        y_trainl, y_testl = yl[train_index], yl[test_index]
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Perform SMOTE algorithm for better data balance
        sm = SMOTE(random_state=2)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
        scaler = preprocessing.StandardScaler().fit(X_train_res)
        X_scaled_train = scaler.transform(X_train_res)
        X_scaled_test = scaler.transform(X_test)
        rfaud.fit(X_scaled_train, y_train_res.ravel())
        X_train_res_ff, y_train_res_ff = sm.fit_resample(X_trainff, y_trainff.ravel())
        scaler = preprocessing.StandardScaler().fit(X_train_res_ff)
        X_scaled_train_ff = scaler.transform(X_train_res_ff)
        X_scaled_test_ff = scaler.transform(X_testff)
        rfall.fit(X_scaled_train_ff, y_train_res_ff.ravel())
        X_train_res_l, y_train_res_l = sm.fit_resample(X_trainl, y_trainl.ravel())
        scaler = preprocessing.StandardScaler().fit(X_train_res_l)
        X_scaled_train_l = scaler.transform(X_train_res_l)
        X_scaled_test_l = scaler.transform(X_testl)
        rfling.fit(X_scaled_train_l, y_train_res_l.ravel())
        preds_fu = all_around_fusion_all_s_2(X_trainff, pd.DataFrame(X_scaled_test_ff),
                        y_trainff, y_testff,
                        X_trainl, pd.DataFrame(X_scaled_test_l) ,
                        y_trainl, y_testl,
                        X_train, pd.DataFrame(X_scaled_test) ,
                        y_train, y_test,
                        rfaud, rfling, rfall)
        print(classification_report(y_test,preds_fu))
        print(accuracy_score(y_test, preds_fu))
        print(f1_score(y_test, preds_fu, average = 'weighted'))
        print(confusion_matrix(y_test,preds_fu))
        result.append(accuracy_score(y_test, preds_fu))
        result_recall.append(recall_score(y_test, preds_fu, average="macro"))
        result_precision.append(precision_score(y_test, preds_fu, average="weighted"))
        result_f1.append(f1_score(y_test, preds_fu, average = 'weighted'))
    return sum(result)/len(result),sum(result_recall)/len(result_recall),sum(result_precision)/len(result_precision),sum(result_f1)/len(result_f1)

all_around_fusion_cv_all_s_2(3,x_all, y_all,x_ling, y_ling,x_audio, y_audio)
'''
Returns:
(0.7410654880714761, 0.6039070589844284, 0.7198464398717775, 0.7088491827964462)
'''

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

def all_around_fusion_cv_all_abburi(k,
                        x_ff, y_ff,
                        xl, yl,
                        x,y):    
    '''Performs k-fold CV of Abburi Decision-Level Fusion.
    Returns: Average classification metrics
    including: weighted precision and F1-Score,
    accuracy and macro recall'''
    rfaud = LogisticRegression(C=1,penalty='l1', verbose=0, solver='liblinear')
    rfall= LogisticRegression(C=1,penalty='l1', verbose=0, solver='liblinear')
    rfling = LogisticRegression(C=1,penalty='l1', verbose=0, solver='liblinear')
    result = []
    result_recall = []
    result_precision = []
    result_f1 = []
    kf = StratifiedKFold(n_splits=k, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(x, y):
        print('Fold no : ', fold_count)
        X_trainff, X_testff = x_ff.iloc[train_index], x_ff.iloc[test_index]
        y_trainff, y_testff = y_ff[train_index], y_ff[test_index]
        X_trainl, X_testl = xl.iloc[train_index], xl.iloc[test_index]
        y_trainl, y_testl = yl[train_index], yl[test_index]
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Perform SMOTE algorithm for better data balance
        sm = SMOTE(random_state=2)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
        scaler = preprocessing.StandardScaler().fit(X_train_res)
        X_scaled_train = scaler.transform(X_train_res)
        X_scaled_test = scaler.transform(X_test)
        rfaud.fit(X_scaled_train, y_train_res.ravel())
        X_train_res_ff, y_train_res_ff = sm.fit_resample(X_trainff, y_trainff.ravel())
        scaler = preprocessing.StandardScaler().fit(X_train_res_ff)
        X_scaled_train_ff = scaler.transform(X_train_res_ff)
        X_scaled_test_ff = scaler.transform(X_testff)
        rfall.fit(X_scaled_train_ff, y_train_res_ff.ravel())
        X_train_res_l, y_train_res_l = sm.fit_resample(X_trainl, y_trainl.ravel())
        scaler = preprocessing.StandardScaler().fit(X_train_res_l)
        X_scaled_train_l = scaler.transform(X_train_res_l)
        X_scaled_test_l = scaler.transform(X_testl)
        rfling.fit(X_scaled_train_l, y_train_res_l.ravel())
        preds_fu = all_around_fusion_all_abburi(X_trainff, pd.DataFrame(X_scaled_test_ff),
                        y_trainff, y_testff,
                        X_trainl, pd.DataFrame(X_scaled_test_l) ,
                        y_trainl, y_testl,
                        X_train, pd.DataFrame(X_scaled_test) ,
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


all_around_fusion_cv_all_abburi(3,x_all, y_all,x_ling, y_ling,x_audio, y_audio)
'''
Returns:
(0.7092006463263948, 0.5903366352109098, 0.6824562012873875, 0.6895237469474024)
'''


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
            y_pred_ling = rfling.predict_proba(np.array(X_test_now_ling).reshape(1, -1))
            actual_pred = int(np.where(y_pred_ling[0] == max(y_pred_ling[0]))[0])
            #print(total)
            print(actual_pred)
            if actual_pred == 0:
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
    rfaud = LogisticRegression(C=1,penalty='l1', verbose=0, solver='liblinear')
    rfall= LogisticRegression(C=1,penalty='l1', verbose=0, solver='liblinear')
    rfling = LogisticRegression(C=1,penalty='l1', verbose=0, solver='liblinear')
    result = []
    result_recall = []
    result_precision = []
    result_f1 = []
    kf = StratifiedKFold(n_splits=k, random_state=None, shuffle = True)
    fold_count = 0
    for train_index, test_index in kf.split(x, y):
        print('Fold no : ', fold_count)
        X_trainff, X_testff = x_ff.iloc[train_index], x_ff.iloc[test_index]
        y_trainff, y_testff = y_ff[train_index], y_ff[test_index]
        X_trainl, X_testl = xl.iloc[train_index], xl.iloc[test_index]
        y_trainl, y_testl = yl[train_index], yl[test_index]
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Perform SMOTE algorithm for better data balance
        sm = SMOTE(random_state=2)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
        scaler = preprocessing.StandardScaler().fit(X_train_res)
        X_scaled_train = scaler.transform(X_train_res)
        X_scaled_test = scaler.transform(X_test)
        rfaud.fit(X_scaled_train, y_train_res.ravel())
        X_train_res_ff, y_train_res_ff = sm.fit_resample(X_trainff, y_trainff.ravel())
        scaler = preprocessing.StandardScaler().fit(X_train_res_ff)
        X_scaled_train_ff = scaler.transform(X_train_res_ff)
        X_scaled_test_ff = scaler.transform(X_testff)
        rfall.fit(X_scaled_train_ff, y_train_res_ff.ravel())
        X_train_res_l, y_train_res_l = sm.fit_resample(X_trainl, y_trainl.ravel())
        scaler = preprocessing.StandardScaler().fit(X_train_res_l)
        X_scaled_train_l = scaler.transform(X_train_res_l)
        X_scaled_test_l = scaler.transform(X_testl)
        rfling.fit(X_scaled_train_l, y_train_res_l.ravel())
        preds_fu = decision_fusion(X_trainff, pd.DataFrame(X_scaled_test_ff),
                        y_trainff, y_testff,
                        X_trainl, pd.DataFrame(X_scaled_test_l) ,
                        y_trainl, y_testl,
                        X_train, pd.DataFrame(X_scaled_test) ,
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


all_around_fusion_cv_all_jia(3,x_all, y_all,x_ling, y_ling,x_audio, y_audio)
'''
Returns:
(0.5696939454424484, 0.6035912803079148, 0.682908453788983, 0.5896674267251147)
'''
