from flask import Flask, render_template, request
import pickle

import nltk
import pandas as pd
import contractions
import string
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# classification models
svm_model = pickle.load(open('model_svm.pkl', 'rb'))
nb_model = pickle.load(open('model_nb.pkl', 'rb'))
lr_model = pickle.load(open('model_lr.pkl', 'rb'))
knn_model = pickle.load(open('model_knn.pkl', 'rb'))
dt_model = pickle.load(open('model_dt.pkl', 'rb'))

# vectorizer
tfidf_vec = pickle.load(open('tfidf_vec.pkl', 'rb'))

@app.route('/', methods=['GET'])

def home():
    return render_template('index.html')

# function to return string based on class
def output_class(c):
    if c == 0:
        return 'Fake News'
    elif c == 1:
        return 'News Is True'
    
# function to return confidence score of prediction
def output_score(s):
    con_score = max(s)
    con_score = str('{:04.3f}'.format(con_score))
    return con_score

# URL removal
def removeURL(text):
    text = re.sub(r'http\S+', '', text)
    return text

# remove punctuation
p = string.punctuation
    
# convert to wordnet pos
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
# lemmatization
wnl = WordNetLemmatizer()

# get model from form input
def getModel(model_name):
    m = svm_model
    if model_name == 'svm':
        m = svm_model
    elif model_name == 'nb':
        m = nb_model
    elif model_name == 'lr':
        m = lr_model
    elif model_name == 'knn':
        m = knn_model
    elif model_name == 'dt':
        m = dt_model
    return m

# text cleaning and prediction
def predict(news, model_selected):
    # create dataframe to store news input
    test_news = {'text': [news]}
    df_test = pd.DataFrame(test_news)

    # text preprocessing
    df_test['text'] = df_test['text'].apply(lambda x: x.lower())
    df_test['text'] = df_test['text'].apply(lambda x: removeURL(x))
    df_test['text'] = df_test['text'].apply(lambda x: [contractions.fix(w) for w in x.split()])
    df_test['text'] = [' '.join(map(str, r)) for r in df_test['text']]
    df_test['text'] = df_test['text'].apply(word_tokenize)
    df_test['text'] = df_test['text'].apply(lambda x: [w for w in x if w not in p])
    stop_words = set(stopwords.words('english'))
    df_test['text'] = df_test['text'].apply(lambda x: [w for w in x if w not in stop_words])
    df_test['text'] = df_test['text'].apply(nltk.tag.pos_tag)
    df_test['text'] = df_test['text'].apply(lambda x: [(w, get_wordnet_pos(t)) for (w, t) in x])
    df_test['text'] = df_test['text'].apply(lambda x: [wnl.lemmatize(w, t) for w, t in x])
    df_test['text'] = [' '.join(map(str, r)) for r in df_test['text']]

    # feature extraction
    news_x_test = df_test['text']
    news_x_test_tfidf = tfidf_vec.transform(news_x_test)

    # prediction using model
    pred_model = model_selected.predict(news_x_test_tfidf)

    # confidence score
    pred_score = model_selected.predict_proba(news_x_test_tfidf)

    return output_class(pred_model[0]), output_score(pred_score[0])

@app.route('/', methods=['POST'])

def webapp():
    select_model = request.form.get('selected_model')
    selected_model = getModel(select_model)
    text = request.form['text']
    prediction, score = predict(text, selected_model)
    return render_template('index.html', selected_model=selected_model, text=text, result=prediction, score=score)

if __name__ == "__main__":
    app.run(debug=True)