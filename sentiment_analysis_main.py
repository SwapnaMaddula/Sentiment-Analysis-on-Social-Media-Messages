# coding: utf-8

# ## Program Assignment - 1

# ### 1. Import Libraries

import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import linear_model 
import warnings
warnings.filterwarnings("ignore")

# ### 2. Training Data

train = pd.read_csv('data/train.dat',sep='\t', dtype={'Text': 'str'})
train_original = train
train['Text'] = train['Text'].apply(lambda x: " ".join(x for x in str(x).split()))
train.head()

# ### 3. PreProcessing

# #### 3.1 Expand all the short forms

contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "I would",
"i'd've": "I would have",
"i'll": "I will",
"i'll've": "I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

def expand(line):
    line = line.lower()
    res = ''
    words = line.split()
    for word in words:
        if(word in contractions):
            res = res+" "+ contractions[word]
        else:
            res = res+" "+word
    return res

train['Text'] = train.apply(lambda row: expand(str(row['Text'])), axis=1)
train.head()

# #### 3.2 Count all the stopwords

nltk_stop_words = stopwords.words('english')
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 
              'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 
              'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 
              'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 
              'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
              'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
              'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
              'all', 'any', 'both', 'each', 'more', 'most', 'other', 'such', 'own', 
              'same', 'so', 'too', 'very', 's', 't', 'just', 'don', 'now']

train['stopwords'] = train['Text'].apply(lambda x: len([x for x in str(x).split() if x in stop_words]))
train[['Text','stopwords']].head()

# #### 3.3 Make the lower case text free of special characters and stop words

train['Text'] = train['Text'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))
train['Text'] = train['Text'].str.replace('[^a-zA-Z\s]','')
train['Text'] = train['Text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))

train.head()

# #### 3.4 Replace all the shortcuts of messaging slang with actual words

shortcuts = pd.read_excel('data/shortforms.xlsx', sheet_name='Sheet1', names =['shortform', 'longform'], header = None)
shortcuts['shortform'] = shortcuts['shortform'].str.lower()
shortcuts['longform'] = shortcuts['longform'].str.lower()
shortcuts.head()

import re
rep = dict(zip(shortcuts.shortform, shortcuts.longform)) #convert into dictionary
def replace(line):
    line = line.lower()
    res = ''
    words = line.split()
    for word in words:
        word = re.sub(r'(.)\1+', r'\1\1', word) 
        if(word in rep):
            res = res+" "+ rep[word]
        else:
            res = res+" "+word
    return res

# #### 3.5 Repeat the preprocessing steps after replacing shortcuts

train['Text'] = train.apply(lambda row: replace(row['Text']), axis=1)
train['Text'] = train['Text'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))
train['Text'] = train['Text'].str.replace('[^a-zA-Z\s]','')
train['Text'] = train['Text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))

train.head()

# #### 3.6 Stem the words to its root word

stemmer = PorterStemmer()
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence
train['Text'] = train['Text'].apply(stemming)

# ### 4. TF-IDF Vectorization

tfidfvec = TfidfVectorizer(sublinear_tf=True, lowercase=True, analyzer='word', stop_words= stop_words,ngram_range=(1,3), min_df=1, norm='l2', use_idf = False)

# ### 5. Seperate the labels

train_vect = tfidfvec.fit(train['Text'])
X_train = train_vect.transform(train['Text'])
y_train = train['sentiment'].values

X_train.shape, y_train.shape

# ### 6. Test Data

test = pd.read_csv('data/test.dat', names =['text'], sep= '\r', header = None, skip_blank_lines=False)
test_original = test
test.head()

# ### 7. Repeat preprocessing on test data

test['text'] = test.apply(lambda row: expand(str(row['text'])), axis=1)
train.head()

test['text'] = test['text'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))
test['text'] = test['text'].str.replace('[^a-zA-Z\s]','')
test['text'] = test['text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
test.head()

test['text'] = test.apply(lambda row: replace(row['text']), axis=1)
test['text'] = test['text'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))
test['text'] = test['text'].str.replace('[^a-zA-Z\s]','')
test['text'] = test['text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
test['text'] = test['text'].apply(stemming)
X_test = tfidfvec.transform(test['text'])
test.head()

# ### 8. Model Selection

# #### 8.1 Linear Support Vector Classification

text_clf =  svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='hinge', 
                          max_iter=1000, multi_class='ovr', penalty='l2', random_state=42, tol=0.0001,verbose=0)
_ = text_clf.fit(X_train, y_train)

# #### 8.2 Logistic Regression

classifier = linear_model.LogisticRegression()

# ### 9. Split training data into train and validation sets

X_tr, X_tst, y_tr, y_tst = train_test_split( X_train, y_train, test_size=0.20, random_state=42)
text_clf.fit(X_tr, y_tr)


classifier.fit(X_tr, y_tr)

# ### 10. Compare the scores by prediction on validation set

predicted = text_clf.predict(X_tst)
from sklearn.metrics import accuracy_score, f1_score
print("Accuracy", accuracy_score(y_tst, predicted))
print("Fscore", f1_score(y_tst, predicted, average='weighted'))

predictions = classifier.predict(X_tst)
print("Accuracy", accuracy_score(y_tst, predictions))
print("Fscore", f1_score(predictions, y_tst, average='weighted'))

# ### 11. Save the best predictions to output file

_ = classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)

np.savetxt('output_012551799.dat', predicted, fmt=['%d'])
predicted

