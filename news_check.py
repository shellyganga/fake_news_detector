import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
recall_score, roc_auc_score, roc_curve, accuracy_score)
import pickle
from nltk.corpus import stopwords
full_train = pd.read_csv("/Users/shellyschwartz/Downloads/train-2.csv")
after_test = pd.read_csv("/Users/shellyschwartz/Downloads/test.csv")

full_train['set'] = 'train'

after_test['set'] = 'test'

concat_df = pd.concat([full_train, after_test])

concat_df.text = concat_df.text.astype(str)

concat_df = concat_df.fillna(' ')

sw = stopwords.words('english')

tfidf = TfidfVectorizer(min_df = 50, stop_words = sw)
tfidf.fit(concat_df['text'])

train_df = concat_df[concat_df['set'] == 'train']
test_df = concat_df[concat_df['set'] == 'test']
print(train_df)
print(test_df)
train, test = train_test_split(train_df, test_size=.4, stratify=train_df.label)

X_train = tfidf.transform(train['text'])
X_test = tfidf.transform(test['text'])

y_train = (train['label'] == 1)
y_test = (test['label'] == 1)


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth = 100, n_estimators = 500, n_jobs = -1)
rf.fit(X_train, y_train)
yhatrf = (rf.predict(X_test))
y_probrf = rf.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_probrf))
print(accuracy_score(y_test, yhatrf))
print(f1_score(y_test, yhatrf))

filename = 'finalized_model.sav'
pickle.dump(rf, open(filename, 'wb'))