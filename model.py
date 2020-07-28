import pandas as pd
import string
import nltk
import re
import pickle
stopwords = nltk.corpus.stopwords.words('english')

ps = nltk.PorterStemmer()

df_train = pd.read_csv('atis_intents_train.csv',header=None)

df_train.columns = ['label','text']

#Create function to remove punctuation, tokenize, remove stopwords, and stem

def clean_text(text):
    text = ''.join([word.lower() for word in text if text not in string.punctuation])
    tokens = re.split('\W+', text)
    text = ' '.join([ps.stem(word) for word in tokens if word not in stopwords])
    return text

df_train['cleaned_text'] = df_train['text'].apply(lambda x : clean_text(x))

from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer()
X = tv.fit_transform(df_train['cleaned_text'])
X = X.toarray()
y = df_train.iloc[:,0].values

pickle.dump(tv,open('cv_transfomr.pkl','wb'))

from sklearn.svm import SVC
svm = SVC()
svm.fit(X,y)

#pickle.dump(svm,open('model.pkl','wb'))

#from sklearn.naive_bayes import BernoulliNB
#nb = BernoulliNB(alpha=0.01)
#nb.fit(X,y)
#pickle.dump(nb,open('model.pkl','wb'))


