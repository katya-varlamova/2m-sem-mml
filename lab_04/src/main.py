import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from pymorphy2 import MorphAnalyzer
import warnings

warnings.filterwarnings("ignore")
morph = MorphAnalyzer()
stop_words = set(stopwords.words('english'))
lemma = True

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    if lemma:
        tokens = [morph.parse(token)[0].normal_form for token in tokens]
    return ' '.join(tokens)

def CountVectorization(X_train, X_test):
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)
    return X_train_counts, X_test_counts
def tfidfVectorization(X_train, X_test):
    tfidf_vectorizer = TfidfVectorizer()
    X_train_counts = tfidf_vectorizer.fit_transform(X_train.apply(preprocess_text))
    X_test_counts = tfidf_vectorizer.transform(X_test.apply(preprocess_text))
    return X_train_counts, X_test_counts

data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

data['label'] = data['label'].map({'ham': 0, 'spam': 1})
X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vecs = ["CountVectorization", "tfidfVectorization"]
i = 0
lemmas = [True]
for vectorization in [CountVectorization, tfidfVectorization]:
    for l in lemmas:
        lemma = l
        #print(l)
        print(vecs[i])
        X_train_counts, X_test_counts = vectorization(X_train, X_test)

        clf = MultinomialNB()
        clf.fit(X_train_counts, y_train)


        y_pred = clf.predict(X_test_counts)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        print("Accuracy: {:.5f}".format(accuracy))
        print("f-мера: {:.5f}".format(f1))
        cm = confusion_matrix(y_test, y_pred)
        print(f"Матрица ошибок:")
        print(cm)
        print()
    i += 1
    
